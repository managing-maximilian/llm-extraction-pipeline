"""Shared helpers — I/O, metadata, prompt/raw saving."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .settings import Settings


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Text extraction ─────────────────────────────────────────────────────────

def get_text(data: dict) -> str:
    """Return the best available text from a pipeline document."""
    return data.get("working_input") or data.get("input", "")


# ── Metadata ────────────────────────────────────────────────────────────────

def preserve_metadata(data: dict) -> dict:
    """Extract pass-through metadata fields from an input document.

    Adjust the keys list to match your project's metadata fields.
    """
    keys = ["input", "source", "pipeline_metadata"]
    meta: dict[str, Any] = {}
    for k in keys:
        if k in data:
            meta[k] = data[k]
    return meta


def build_pipeline_metadata(settings: "Settings", step_name: str) -> Dict[str, Any]:
    """Build a metadata dict capturing the current pipeline run config."""
    llm = settings.get_llm_for_step(step_name)
    meta: Dict[str, Any] = {
        "model_provider": llm.provider,
        "prompt_language": settings.prompt_language,
        "output_language": settings.output_language,
        "temperature": llm.temperature,
        "step": step_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if llm.provider == "local":
        meta["model_preset"] = llm.model_preset
        meta["model_repo_id"] = llm.repo_id
        meta["backend"] = "llama-server"
    else:
        meta["model_name"] = llm.model_name
    return meta


# ── Prompt / raw-output saving ──────────────────────────────────────────────

def save_prompt(
    settings: "Settings",
    step_name: str,
    file_stem: str,
    system_prompt: str,
    user_message: str,
    suffix: str = "",
) -> None:
    """Save system + user prompt to <output_dir>/<step>/prompts/<file>_prompt.txt."""
    if not settings.save_prompts:
        return
    prompt_dir = Path(settings.get_output_dir(step_name)) / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{file_stem}{suffix}_prompt.txt"
    with open(prompt_dir / fname, "w", encoding="utf-8") as f:
        f.write("=== SYSTEM MESSAGE ===\n\n")
        f.write(system_prompt)
        f.write("\n\n=== USER MESSAGE ===\n\n")
        f.write(user_message)


def save_raw(
    settings: "Settings",
    step_name: str,
    file_stem: str,
    content: str,
    *,
    reasoning: Optional[str] = None,
    suffix: str = "",
) -> None:
    """Save raw LLM response to <output_dir>/<step>/raw_outputs/<file>_raw.txt."""
    if not settings.save_raw_output:
        return
    raw_dir = Path(settings.get_output_dir(step_name)) / "raw_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{file_stem}{suffix}_raw.txt"
    with open(raw_dir / fname, "w", encoding="utf-8") as f:
        if reasoning:
            f.write("=== REASONING ===\n\n")
            f.write(reasoning)
            f.write("\n\n=== CONTENT ===\n\n")
        f.write(content)
