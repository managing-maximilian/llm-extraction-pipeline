"""
Pipeline runner — the single entry point.

Usage:
    python -m pipeline                           # run with config.yaml defaults
    python -m pipeline --config my_config.yaml   # use a different config
    python -m pipeline --steps step1 step2       # run specific steps
    python -m pipeline --input-dir data/input/v2 # override input dir
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Callable

from .settings import Settings, load_settings
from .helpers import load_json, save_json, build_pipeline_metadata
from .provider import stop_server
from .steps import step1

# ── Step registry ────────────────────────────────────────────────────────────
# Add new steps here as you create them.

STEP_REGISTRY: Dict[str, Callable] = {
    "step1": step1.run,
    # "step2": step2.run,
}

# Maps each step to its predecessor (where it reads input from).
# The first step in the chain reads from settings.input_dir.
_STEP_INPUT_MAP: Dict[str, str] = {
    # "step2": "step1",
}


# ── Console + file logging ──────────────────────────────────────────────────

class _Tee:
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, original, log_path: Path):
        self._orig = original
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        self._orig.write(text)
        self._orig.flush()
        self._log.write(text)

    def flush(self):
        self._orig.flush()
        self._log.flush()

    def close(self):
        self._log.close()

    def __getattr__(self, name):
        return getattr(self._orig, name)


# ── Input directory resolution ──────────────────────────────────────────────

def _get_input_dir(step_name: str, settings: Settings, *, allow_fallback: bool = False) -> Path:
    """Determine where a step reads its input files from.

    Walks _STEP_INPUT_MAP to find the predecessor's output dir.
    Falls back to settings.input_dir for the first step or when allow_fallback=True.
    """
    prev_step = _STEP_INPUT_MAP.get(step_name)
    if prev_step:
        prev_dir = Path(settings.get_output_dir(prev_step))
        if prev_dir.exists() and any(prev_dir.glob("*.json")):
            return prev_dir
        if allow_fallback:
            return _get_input_dir(prev_step, settings, allow_fallback=True)
        raise FileNotFoundError(
            f"{step_name}: expected input from '{prev_step}' at {prev_dir}, "
            f"but no JSON files found. Did the previous step fail?"
        )
    return Path(settings.input_dir)


# ── Execution modes ─────────────────────────────────────────────────────────

def _run_step_by_step(settings: Settings) -> None:
    """Process all files for step N, then all files for step N+1, etc."""
    first_step = settings.steps_to_run[0] if settings.steps_to_run else None

    for step_name in settings.steps_to_run:
        step_fn = STEP_REGISTRY.get(step_name)
        if step_fn is None:
            print(f"Unknown step: {step_name}, skipping")
            continue

        input_dir = _get_input_dir(step_name, settings, allow_fallback=(step_name == first_step))
        output_dir = Path(settings.get_output_dir(step_name))
        input_files = sorted(input_dir.glob("*.json"))

        print(f"\n{'='*60}")
        print(f"  {step_name}  |  {len(input_files)} files  |  {input_dir} -> {output_dir}")
        print(f"{'='*60}")

        for idx, input_file in enumerate(input_files, 1):
            print(f"  [{idx}/{len(input_files)}] {input_file.name}")
            try:
                data = load_json(input_file)
                result = step_fn(data, settings, file_stem=input_file.stem)

                result["pipeline_metadata"] = build_pipeline_metadata(settings, step_name)
                save_json(result, output_dir / input_file.name)

            except Exception as exc:
                print(f"    ERROR [{step_name} / {input_file.name}]: {type(exc).__name__}: {exc}")
                print(traceback.format_exc())
                if not settings.continue_on_error:
                    raise


def _run_file_by_file(settings: Settings) -> None:
    """Process all steps for file 1, then all steps for file 2, etc."""
    input_files = sorted(Path(settings.input_dir).glob("*.json"))

    for idx, input_file in enumerate(input_files, 1):
        print(f"\n{'='*60}")
        print(f"  File [{idx}/{len(input_files)}]: {input_file.name}")
        print(f"{'='*60}")

        data = load_json(input_file)

        for step_name in settings.steps_to_run:
            step_fn = STEP_REGISTRY.get(step_name)
            if step_fn is None:
                continue

            print(f"  > {step_name}")
            try:
                result = step_fn(data, settings, file_stem=input_file.stem)

                result["pipeline_metadata"] = build_pipeline_metadata(settings, step_name)
                output_dir = Path(settings.get_output_dir(step_name))
                save_json(result, output_dir / input_file.name)

                data = result  # feed this step's output as next step's input

            except Exception as exc:
                print(f"    ERROR [{step_name} / {input_file.name}]: {type(exc).__name__}: {exc}")
                print(traceback.format_exc())
                if not settings.continue_on_error:
                    raise
                break


# ── Public entry point ──────────────────────────────────────────────────────

def run_pipeline(settings: Settings | None = None) -> None:
    if settings is None:
        settings = load_settings()

    # Log file
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = Path(settings.output_dir) / f"pipeline_{ts}.log"
    tee_out = _Tee(sys.stdout, log_path)
    tee_err = _Tee(sys.stderr, log_path)
    sys.stdout = tee_out  # type: ignore[assignment]
    sys.stderr = tee_err  # type: ignore[assignment]

    try:
        print(f"Log:      {log_path}")
        print(f"Pipeline: {settings.steps_to_run}")
        print(f"Input:    {settings.input_dir}")
        print(f"Output:   {settings.output_dir}")

        llm = settings.llm
        if llm.provider == "local":
            label = llm.model_preset or llm.repo_id or llm.model_path or "unknown"
            print(f"LLM:      local / llama-server ({label})")
        else:
            print(f"LLM:      {llm.provider} ({llm.model_name})")

        if settings.execution_mode == "file_by_file":
            _run_file_by_file(settings)
        else:
            _run_step_by_step(settings)

        print("\nDone.")
    finally:
        stop_server()
        sys.stdout = tee_out._orig
        sys.stderr = tee_err._orig
        tee_out.close()
        tee_err.close()


def main():
    parser = argparse.ArgumentParser(description="LLM Extraction Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--steps", nargs="*", help="Steps to run (overrides config)")
    parser.add_argument("--input-dir", help="Override input directory")
    parser.add_argument("--output-dir", help="Override output base directory")
    parser.add_argument("--mode", choices=["step_by_step", "file_by_file"], help="Execution mode")
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.steps:
        settings.steps_to_run = args.steps
    if args.input_dir:
        settings.input_dir = args.input_dir
    if args.output_dir:
        settings.output_dir = args.output_dir
    if args.mode:
        settings.execution_mode = args.mode

    run_pipeline(settings)


if __name__ == "__main__":
    main()
