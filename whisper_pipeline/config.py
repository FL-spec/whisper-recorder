"""
PipelineConfig — the single source of truth for all pipeline parameters.

Loaded from config.yaml on startup. If the file is missing, invalid, or
incomplete, a clear Pydantic ValidationError is raised before any model
is loaded or any processing begins.

Nothing in the pipeline is hardcoded — every parameter flows through here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────────────────────────────────────

class WhisperConfig(BaseModel):
    model: str = Field("large-v3", description="faster-whisper model name or HuggingFace repo ID.")
    language: str = Field("pt", description="Target language code (ISO 639-1, e.g. 'pt', 'en').")
    device: Literal["cpu", "cuda", "auto"] = Field("cpu")
    compute_type: str = Field("int8", description="CTranslate2 quantisation (int8, float16, …).")
    beam_size: int = Field(5, ge=1, le=20)
    vad_min_silence_ms: int = Field(
        500,
        ge=100,
        le=5000,
        description="Minimum silence duration in ms for VAD filter.",
    )


class DiarizationConfig(BaseModel):
    enabled: bool = Field(True, description="Set to false to skip diarization entirely.")
    device: Literal["cpu", "cuda"] = Field("cpu")
    hf_token: str = Field(
        default="",
        description="HuggingFace token for pyannote model download. Falls back to HF_TOKEN env var.",
    )
    min_speakers: Optional[int] = Field(None, ge=1, description="Hint to Pyannote: minimum speakers.")
    max_speakers: Optional[int] = Field(None, ge=1, description="Hint to Pyannote: maximum speakers.")

    @model_validator(mode="after")
    def resolve_hf_token(self) -> "DiarizationConfig":
        """If hf_token is blank, try the environment variable."""
        if not self.hf_token:
            self.hf_token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                or ""
            )
        return self


class LLMConfig(BaseModel):
    provider: Literal["ollama"] = Field("ollama")
    model: str = Field("llama3.1:8b", description="Ollama model tag, e.g. 'llama3.1:8b', 'qwen2.5:7b'.")
    base_url: str = Field("http://localhost:11434", description="Ollama server base URL.")
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    request_timeout: int = Field(300, ge=10, description="Per-request timeout in seconds.")
    num_ctx: int = Field(
        8192,
        ge=512,
        description=(
            "Context window size passed to Ollama. The Ollama default (2048) is too "
            "small for lecture blocks. 8192 is safe for llama3.1:8b on CPU/MPS."
        ),
    )


class PipelineSettings(BaseModel):
    block_duration_minutes: int = Field(
        5,
        ge=1,
        le=60,
        description="Duration in minutes of each block fed to the LLM (map phase). 5 min ≈ 1k tokens, safe for 8K context.",
    )
    max_transcript_chars: int = Field(
        6000,
        ge=500,
        description=(
            "Hard cap on the number of transcript characters sent per LLM block. "
            "Prevents context overflow on unusually dense speech. ~6000 chars ≈ 1200 words ≈ 1600 tokens."
        ),
    )
    output_format: Literal["markdown", "plaintext", "json"] = Field("markdown")
    output_dir: str = Field("outputs", description="Directory where all output files are written.")
    prompt_dir: str = Field("prompts", description="Directory containing prompt template .txt files.")
    save_intermediates: bool = Field(
        False,
        description="If true, saves intermediate JSON files (word tokens, speaker segments, aligned segments).",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Top-level config
# ──────────────────────────────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)

    glossary: list[str] = Field(
        default_factory=list,
        description=(
            "Optional domain-specific vocabulary. "
            "Terms here are included in LLM prompts to reduce misrecognition. "
            "The pipeline never fails if this list is empty."
        ),
    )

    @field_validator("glossary", mode="before")
    @classmethod
    def glossary_none_to_empty(cls, v):
        """Allow `glossary: null` in YAML without breaking."""
        return v if v is not None else []


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config.yaml") -> PipelineConfig:
    """
    Load and validate config.yaml. Raises ValueError or ValidationError on
    any problem — before any model is loaded.

    Also loads .env so that HF_TOKEN etc. are in the environment.
    """
    from dotenv import load_dotenv

    load_dotenv()  # always load .env first so env vars are available for field resolution

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path.resolve()}\n"
            "Copy config.yaml.example to config.yaml and edit it."
        )

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    config = PipelineConfig.model_validate(raw)

    # Ensure output and prompt directories exist
    Path(config.pipeline.output_dir).mkdir(parents=True, exist_ok=True)

    return config
