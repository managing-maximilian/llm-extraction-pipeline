"""
Step 1 — Example extraction step.

Each step follows the same pattern:
1. Get text from input data
2. Build prompts
3. Call LLM
4. Parse + validate response
5. Return enriched output dict
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..provider import call_llm
from ..parser import parse_and_validate
from ..prompts.step1 import build_system_prompt, build_user_message
from ..models.step1 import Step1Output
from ..helpers import preserve_metadata, get_text, save_prompt, save_raw

if TYPE_CHECKING:
    from ..settings import Settings

STEP_NAME = "step1"


def run(input_data: dict, settings: "Settings", **_kw) -> dict:
    file_stem: str = _kw.get("file_stem", "")
    llm_cfg = settings.get_llm_for_step(STEP_NAME)
    lang = settings.prompt_language

    # 1. Get the text to process
    text = get_text(input_data)

    # 2. Build prompts
    custom = settings.get_custom_instructions(STEP_NAME)
    system_prompt = build_system_prompt(output_language=settings.output_language, lang=lang)
    if custom:
        system_prompt += f"\n\n{custom}"
    user_message = build_user_message(text, lang=lang)

    # 3. Save prompts (for debugging / reproducibility)
    save_prompt(settings, STEP_NAME, file_stem, system_prompt, user_message)

    # 4. Call LLM
    reasoning, content = call_llm(llm_cfg, system_prompt, user_message)

    # 5. Save raw output
    save_raw(settings, STEP_NAME, file_stem, content, reasoning=reasoning)

    # 6. Parse + validate
    output = parse_and_validate(
        content, Step1Output, settings,
        step_name=STEP_NAME,
        file_stem=file_stem,
        original_system=system_prompt,
        original_user=user_message,
    )

    # 7. Build result — preserve input metadata + add step output
    result = preserve_metadata(input_data)
    result["persons"] = [p.model_dump() for p in output.persons]
    result["events"] = [e.model_dump() for e in output.events]
    return result
