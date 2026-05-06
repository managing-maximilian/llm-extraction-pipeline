"""
Output parser — extract, validate, and optionally auto-fix LLM responses.

Workflow:
1. Extract JSON (markdown fences, bare JSON objects/arrays)
2. Validate against a Pydantic model
3. On failure + fixer enabled: ask the LLM to repair the JSON and retry once
"""

from __future__ import annotations

import json
from typing import TypeVar, Type, TYPE_CHECKING

from pydantic import ValidationError

from .provider import extract_json_from_text, call_llm
from .helpers import save_raw, save_prompt

if TYPE_CHECKING:
    from .settings import Settings

T = TypeVar("T")


def _create_fixing_prompt(malformed_text: str, model_class: type) -> tuple[str, str]:
    """Build system + user messages for the fixer LLM call."""
    schema = model_class.model_json_schema()
    system = (
        "You are an expert at fixing malformed JSON. Your task is to convert "
        "the text I provide into valid JSON that can be parsed by Python's "
        "json.loads() function and validated by Pydantic.\n\n"
        f"Expected schema:\n{json.dumps(schema, indent=2)}\n\n"
        "Fix any syntax errors and ensure the structure matches the schema. "
        "Return ONLY the fixed JSON without any explanation."
    )
    user = (
        "Fix this malformed JSON so it can be parsed and validated:\n\n"
        f"{malformed_text}\n\n"
        "Return ONLY the fixed JSON."
    )
    return system, user


def _create_schema_echo_prompt(
    model_class: type,
    original_system: str = "",
    original_user: str = "",
) -> tuple[str, str]:
    """Build prompts for when the LLM returned the schema instead of data."""
    schema_hint = json.dumps(model_class.model_json_schema(), indent=2)

    if original_system and original_user:
        system = (
            original_system + "\n\n"
            "CRITICAL: Your previous attempt was WRONG — you returned the JSON Schema "
            "definition instead of actual data. Do NOT return the schema. "
            "Return a JSON object with REAL data conforming to the schema."
        )
        user = original_user
    else:
        system = (
            "You returned the JSON Schema definition instead of actual data. "
            "I need you to produce real output data.\n\n"
            f"Expected schema:\n{schema_hint}\n\n"
            "Return ONLY a JSON object with real data matching this schema. "
            "Do NOT return the schema itself."
        )
        user = "Return ONLY the correct JSON output with real data."
    return system, user


def fix_with_llm(
    malformed_text: str,
    model_class: type,
    settings: "Settings",
    step_name: str = "",
    file_stem: str = "",
) -> str:
    """Ask the LLM to repair malformed JSON.

    max_tokens and reasoning_budget are taken from the LLM config (same as the
    main pipeline calls — one setting for all).
    """
    fixer_cfg = settings.fixer
    llm_cfg = fixer_cfg.llm if fixer_cfg.llm is not None else settings.llm

    system_prompt, user_message = _create_fixing_prompt(malformed_text, model_class)

    if step_name:
        save_prompt(settings, step_name, file_stem, system_prompt, user_message, suffix="_fixer")

    print("    Attempting to fix malformed output with LLM ...")
    _reasoning, fixed_content = call_llm(
        llm_cfg,
        system_prompt,
        user_message,
        temperature=fixer_cfg.temperature,
    )
    return fixed_content


def parse_and_validate(
    content: str,
    model_class: "Type[T]",
    settings: "Settings",
    *,
    step_name: str = "",
    file_stem: str = "",
    original_system: str = "",
    original_user: str = "",
) -> "T":
    """Parse LLM content into a validated Pydantic model instance.

    1. Extract JSON from content
    2. json.loads + model_class.model_validate
    3. On schema-echo: retry with warning
    4. On failure + fixer enabled: fix_with_llm, retry once
    """
    if not content or not content.strip():
        raise ValueError(f"[{step_name}/{file_stem}] LLM returned empty content — cannot parse.")

    json_str = extract_json_from_text(content)

    try:
        data = json.loads(json_str)

        # Detect schema-echo (LLM returned the JSON Schema definition)
        if isinstance(data, dict) and "$defs" in data:
            if not settings.fixer.enabled:
                raise ValueError("LLM returned the JSON Schema definition instead of conforming data.")
            print("    Schema-echo detected, retrying ...")
            fixer_cfg = settings.fixer
            llm_cfg = fixer_cfg.llm if fixer_cfg.llm is not None else settings.llm
            sys_prompt, usr_msg = _create_schema_echo_prompt(model_class, original_system, original_user)
            _reasoning, retry_content = call_llm(
                llm_cfg, sys_prompt, usr_msg,
                temperature=fixer_cfg.temperature,
            )
            if step_name:
                save_raw(settings, step_name, file_stem, retry_content, suffix="_schema_retry")
            retry_json = extract_json_from_text(retry_content)
            data = json.loads(retry_json)
            return model_class.model_validate(data)

        return model_class.model_validate(data)

    except (json.JSONDecodeError, ValidationError) as exc:
        if not settings.fixer.enabled:
            raise

        print(f"    Parse/validation failed ({type(exc).__name__}): {exc}")
        print(f"    Extracted JSON snippet: {json_str[:300]!r}")

        fixed_content = fix_with_llm(content, model_class, settings, step_name=step_name, file_stem=file_stem)

        if step_name:
            save_raw(settings, step_name, file_stem, fixed_content, suffix="_fixed")

        fixed_json = extract_json_from_text(fixed_content)
        data = json.loads(fixed_json)
        return model_class.model_validate(data)
