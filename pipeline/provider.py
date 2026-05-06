"""
LLM provider — direct HTTP calls to llama-server / OpenAI / Anthropic.

For local models a llama-server process is started automatically and
accessed via its OpenAI-compatible /v1/chat/completions endpoint.
"""

from __future__ import annotations

import atexit
import json
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import LLMConfig


# ── llama-server process management ─────────────────────────────────────────

_server_proc: subprocess.Popen | None = None
_server_model_key: str | None = None


def _model_key(config: "LLMConfig") -> str:
    return config.model_path or f"{config.repo_id}::{config.filename}"


def _resolve_gguf_path(config: "LLMConfig") -> str:
    """Return the local path to the GGUF file (downloads from HF if needed)."""
    if config.model_path:
        return config.model_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=config.repo_id, filename=config.filename)


def _wait_for_server(port: int, timeout: float = 120.0) -> None:
    """Block until the llama-server /health endpoint returns ok."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                body = resp.read().decode()
                if '"status":"ok"' in body or resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server not ready on port {port} within {timeout}s")


def start_server(config: "LLMConfig") -> None:
    """Start (or reuse) a llama-server subprocess for the given config."""
    global _server_proc, _server_model_key

    key = _model_key(config)
    if _server_model_key == key and _server_proc is not None and _server_proc.poll() is None:
        return

    stop_server()

    gguf_path = _resolve_gguf_path(config)
    cmd = [
        config.llama_server_bin,
        "-m", gguf_path,
        "-c", str(config.n_ctx),
        "-ngl", str(config.n_gpu_layers),
        "--port", str(config.llama_server_port),
        "--host", "127.0.0.1",
        "--reasoning-format", "deepseek",
        "--reasoning-budget", str(config.reasoning_budget),
    ]

    print(f"\n  Starting llama-server: {' '.join(cmd)}")
    _server_proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    _server_model_key = key
    atexit.register(stop_server)

    print(f"  Waiting for llama-server on port {config.llama_server_port} ...")
    _wait_for_server(config.llama_server_port)
    print(f"  llama-server ready (PID {_server_proc.pid}).\n")


def stop_server() -> None:
    """Gracefully terminate the running llama-server, if any."""
    global _server_proc, _server_model_key
    if _server_proc is None:
        return
    if _server_proc.poll() is None:
        print(f"\n  Stopping llama-server (PID {_server_proc.pid}) ...")
        _server_proc.send_signal(signal.SIGTERM)
        try:
            _server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
            _server_proc.wait()
        print("  llama-server stopped.")
    _server_proc = None
    _server_model_key = None


# ── JSON extraction helper ──────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def extract_json_from_text(text: str) -> str:
    """Extract JSON from the model's free-form text response.

    Strips markdown fences, reasoning blocks, and leading/trailing noise.
    """
    cleaned = text.strip()

    # Strip leaked <think> blocks (deepseek/qwen reasoning format)
    think_pos = cleaned.rfind("</think>")
    if think_pos != -1:
        cleaned = cleaned[think_pos + len("</think>"):].strip()

    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        return fence_match.group(1).strip()

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = cleaned.find(start_char)
        if start == -1:
            continue
        end = cleaned.rfind(end_char)
        if end > start:
            return cleaned[start : end + 1]

    return cleaned


# ── LLM call ────────────────────────────────────────────────────────────────

def _build_messages(
    config: "LLMConfig",
    system_prompt: str,
    user_message: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        if config.provider != "local" or config.supports_system_prompt:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    return messages


def _call_local(
    config: "LLMConfig",
    messages: List[Dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_budget: int | None = None,
) -> Tuple[Optional[str], str]:
    start_server(config)
    url = f"http://127.0.0.1:{config.llama_server_port}/v1/chat/completions"
    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature if temperature is not None else config.temperature,
        "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
    }
    if reasoning_budget is not None:
        payload["reasoning_budget"] = reasoning_budget
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())

    msg = body["choices"][0]["message"]
    reasoning = msg.get("reasoning_content") or None
    content = msg.get("content") or ""

    if not content:
        finish_reason = body["choices"][0].get("finish_reason", "unknown")
        print(f"    WARNING: llama-server returned empty content (finish_reason={finish_reason!r})")
        if reasoning:
            print(f"    Model spent all tokens on thinking ({len(reasoning)} chars of reasoning, 0 content)")
        print(f"    Full response (truncated): {json.dumps(body, indent=2)[:500]}")

    return reasoning, content


def _call_openai(
    config: "LLMConfig",
    messages: List[Dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Tuple[Optional[str], str]:
    from openai import OpenAI
    client = OpenAI(api_key=config.api_key)
    response = client.chat.completions.create(
        model=config.model_name,
        messages=messages,
        temperature=temperature if temperature is not None else config.temperature,
        max_tokens=max_tokens if max_tokens is not None else config.max_tokens,
    )
    return None, response.choices[0].message.content


def _call_anthropic(
    config: "LLMConfig",
    messages: List[Dict[str, str]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Tuple[Optional[str], str]:
    import anthropic
    client = anthropic.Anthropic(api_key=config.api_key)

    system_message = ""
    user_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            user_messages.append(msg)

    kwargs: Dict[str, Any] = {
        "model": config.model_name,
        "messages": user_messages,
        "temperature": temperature if temperature is not None else config.temperature,
        "max_tokens": max_tokens if max_tokens is not None else config.max_tokens,
    }
    if system_message:
        kwargs["system"] = system_message

    response = client.messages.create(**kwargs)
    return None, response.content[0].text


def call_llm(
    config: "LLMConfig",
    system_prompt: str,
    user_message: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_budget: int | None = None,
) -> Tuple[Optional[str], str]:
    """Call the LLM and return (reasoning, content).

    reasoning is the model's chain-of-thought (None when not available).
    content is the actual answer.
    """
    messages = _build_messages(config, system_prompt, user_message)

    if config.provider == "local":
        return _call_local(config, messages, temperature=temperature, max_tokens=max_tokens, reasoning_budget=reasoning_budget)
    if config.provider == "openai":
        return _call_openai(config, messages, temperature=temperature, max_tokens=max_tokens)
    if config.provider == "anthropic":
        return _call_anthropic(config, messages, temperature=temperature, max_tokens=max_tokens)

    raise ValueError(f"Unknown LLM provider: {config.provider!r}")
