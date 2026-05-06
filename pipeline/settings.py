"""
Pipeline settings — loaded from config.yaml, overridable via CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ── Local model presets ──────────────────────────────────────────────────────
# Add your own presets here.  Set  llm.model_preset  in config.yaml.

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen3.5-35b": {
        "repo_id": "bartowski/Qwen_Qwen3.5-35B-A3B-GGUF",
        "filename": "Qwen_Qwen3.5-35B-A3B-Q4_K_S.gguf",
        "supports_system_prompt": True,
    },
    "qwen3-14b": {
        "repo_id": "bartowski/Qwen_Qwen3-14B-GGUF",
        "filename": "Qwen_Qwen3-14B-Q6_K_L.gguf",
        "supports_system_prompt": True,
    },
    "qwen3-32b": {
        "repo_id": "bartowski/Qwen_Qwen3-32B-GGUF",
        "filename": "Qwen_Qwen3-32B-Q4_K_M.gguf",
        "supports_system_prompt": True,
    },
    "gemma-3-27b": {
        "repo_id": "bartowski/google_gemma-3-27b-it-qat-GGUF",
        "filename": "google_gemma-3-27b-it-qat-Q5_K_L.gguf",
        "supports_system_prompt": False,
    },
    "mistral-24b-instruct": {
        "repo_id": "bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
        "filename": "Mistral-Small-24B-Instruct-2501-Q6_K_L.gguf",
        "supports_system_prompt": True,
    },
    "ministral-3-14b-reasoning": {
        "repo_id": "bartowski/mistralai_Ministral-3-14B-Reasoning-2512-GGUF",
        "filename": "mistralai_Ministral-3-14B-Reasoning-2512-Q8_0.gguf",
        "supports_system_prompt": True,
    },
    "deepseek-r1-distill-qwen-32b": {
        "repo_id": "bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF",
        "filename": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_L.gguf",
        "supports_system_prompt": True,
    },
}


# ── Pydantic config models ──────────────────────────────────────────────────

class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "local"

    # Local model
    model_preset: Optional[str] = "qwen3.5-35b"
    repo_id: Optional[str] = None
    filename: Optional[str] = None
    supports_system_prompt: Optional[bool] = None
    model_path: Optional[str] = None

    n_ctx: int = 65536
    n_gpu_layers: int = -1

    # llama-server
    llama_server_bin: str = "/usr/local/bin/llama-server"
    llama_server_port: int = 8180

    # Cloud providers
    model_name: Optional[str] = None
    api_key: Optional[str] = None

    # Generation
    temperature: float = 0.0
    max_tokens: int = 32768
    reasoning_budget: int = 16384

    @model_validator(mode="after")
    def _apply_preset(self) -> "LLMConfig":
        if self.provider != "local" or self.model_path:
            if self.supports_system_prompt is None:
                self.supports_system_prompt = True
            return self

        if self.model_preset:
            preset = MODEL_PRESETS.get(self.model_preset)
            if preset is None:
                raise ValueError(
                    f"Unknown model_preset {self.model_preset!r}. "
                    f"Available: {', '.join(MODEL_PRESETS)}"
                )
            if self.repo_id is None:
                self.repo_id = preset["repo_id"]
            if self.filename is None:
                self.filename = preset["filename"]
            if self.supports_system_prompt is None:
                self.supports_system_prompt = preset["supports_system_prompt"]
        else:
            if not self.repo_id or not self.filename:
                raise ValueError("Either model_preset, model_path, or repo_id+filename must be set")
            if self.supports_system_prompt is None:
                self.supports_system_prompt = True

        return self


class FixerConfig(BaseModel):
    """Auto-fixer for malformed LLM responses."""

    enabled: bool = True
    llm: Optional[LLMConfig] = None
    temperature: float = 0.0


class StepConfig(BaseModel):
    """Per-step overrides (all optional)."""

    llm: Optional[LLMConfig] = None
    custom_instructions: str = ""
    output_dir: Optional[str] = None


class Settings(BaseModel):
    """Main pipeline settings."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    fixer: FixerConfig = Field(default_factory=FixerConfig)

    # Directories
    input_dir: str = "data/input"
    output_dir: str = "data/output"

    # Pipeline
    steps_to_run: List[str] = Field(default=["step1"])
    execution_mode: str = "step_by_step"

    # Processing
    output_language: str = "German"
    prompt_language: str = "en"
    continue_on_error: bool = True
    save_raw_output: bool = True
    save_prompts: bool = True
    verbose: bool = True

    # Per-step overrides — add entries as you add steps
    step1: StepConfig = Field(default_factory=StepConfig)
    # step2: StepConfig = Field(default_factory=StepConfig)

    def get_llm_for_step(self, step_name: str) -> LLMConfig:
        step_cfg: StepConfig = getattr(self, step_name, StepConfig())
        return step_cfg.llm if step_cfg.llm is not None else self.llm

    def get_output_dir(self, step_name: str) -> str:
        step_cfg: StepConfig = getattr(self, step_name, StepConfig())
        if step_cfg.output_dir:
            return step_cfg.output_dir
        return f"{self.output_dir}/{step_name}"

    def get_custom_instructions(self, step_name: str) -> str:
        step_cfg: StepConfig = getattr(self, step_name, StepConfig())
        return step_cfg.custom_instructions


def load_settings(config_path: str = "config.yaml") -> Settings:
    """Load settings from a YAML config file."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Settings(**raw)
    return Settings()
