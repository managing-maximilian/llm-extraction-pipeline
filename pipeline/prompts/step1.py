"""
Prompt builder for Step 1.

Each step has its own prompt file with build_system_prompt() and build_user_message().
Customize the prompts for your use case — the rest of the pipeline stays the same.
"""

import json

from ..models.step1 import Step1Output

OUTPUT_FORMAT_EXAMPLE = Step1Output.get_example_output()


def build_system_prompt(output_language: str = "German", lang: str = "en") -> str:
    """Build the system prompt.

    Args:
        output_language: Language the LLM should write its answer in.
        lang: Prompt template language ("en" or "de").
    """
    # ── Add more languages as needed ─────────────────────────────────────
    if lang == "de":
        return f"""Du bist ein sorgfaeltiger Informations-Extractor.

Aufgabe:
Extrahiere strukturierte Informationen aus dem gegebenen Text.

1. PERSONEN: Liste alle namentlich genannten Personen auf. Wenn ein Titel,
   eine Funktion oder ein Beruf genannt ist, gib ihn als 'role' an, sonst null.
2. EREIGNISSE: Liste alle beschriebenen Handlungen oder Ereignisse als kurze,
   sachliche Beschreibungen auf. Gib bei jedem Ereignis die beteiligten
   Personen unter 'actors' an.

Erfinde keine Informationen, die nicht im Text stehen. Schreibe alle Werte
auf {output_language}.

Antworte ausschliesslich mit gueltigem JSON in genau diesem Format:
{json.dumps(OUTPUT_FORMAT_EXAMPLE, indent=2, ensure_ascii=False)}"""

    # ── English (default) ────────────────────────────────────────────────
    return f"""You are a careful information extractor.

Task:
Extract structured information from the given text.

1. PERSONS: List every named person. If a title, role or profession is given,
   record it as 'role'; otherwise use null.
2. EVENTS: List every described action or event as a short, factual
   description. For each event, list the involved persons under 'actors'.

Do not invent information that is not in the text. Write all values in
{output_language}.

You must respond with valid JSON in this exact format:
{json.dumps(OUTPUT_FORMAT_EXAMPLE, indent=2, ensure_ascii=False)}"""


def build_user_message(text: str, lang: str = "en") -> str:
    """Build the user message containing the text to extract from."""
    if lang == "de":
        return f"""Extrahiere Personen und Ereignisse aus dem folgenden Text:

{text}

Gib NUR die JSON-Ausgabe zurueck."""

    return f"""Extract persons and events from the following text:

{text}

Return ONLY the JSON output."""
