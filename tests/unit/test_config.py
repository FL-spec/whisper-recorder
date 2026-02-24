"""
Unit tests for PipelineConfig validation.

Tests that:
- Valid config parses correctly.
- Missing required fields use sensible defaults.
- Invalid enum values raise ValidationError.
- Invalid numeric ranges raise ValidationError.
- HF token is resolved from environment when not in config.
- Glossary can be null, empty, or a list.
"""
from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from whisper_pipeline.config import PipelineConfig


class TestDefaultConfig:
    def test_empty_dict_uses_all_defaults(self):
        cfg = PipelineConfig.model_validate({})
        assert cfg.whisper.model == "large-v3"
        assert cfg.whisper.language == "pt"
        assert cfg.whisper.device == "cpu"
        assert cfg.diarization.enabled is True
        assert cfg.llm.model == "llama3.1:8b"
        assert cfg.pipeline.block_duration_minutes == 5
        assert cfg.pipeline.output_format == "markdown"
        assert cfg.glossary == []

    def test_partial_override(self):
        cfg = PipelineConfig.model_validate({"whisper": {"language": "en", "beam_size": 3}})
        assert cfg.whisper.language == "en"
        assert cfg.whisper.beam_size == 3
        assert cfg.whisper.model == "large-v3"  # default preserved


class TestWhisperConfig:
    def test_valid_devices(self):
        for device in ("cpu", "cuda", "auto"):
            cfg = PipelineConfig.model_validate({"whisper": {"device": device}})
            assert cfg.whisper.device == device

    def test_invalid_device(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"whisper": {"device": "gpu"}})

    def test_beam_size_bounds(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"whisper": {"beam_size": 0}})
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"whisper": {"beam_size": 21}})

    def test_vad_min_silence_bounds(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"whisper": {"vad_min_silence_ms": 50}})
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"whisper": {"vad_min_silence_ms": 6000}})


class TestDiarizationConfig:
    def test_disabled(self):
        cfg = PipelineConfig.model_validate({"diarization": {"enabled": False}})
        assert cfg.diarization.enabled is False

    def test_hf_token_from_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test_token_from_env")
        cfg = PipelineConfig.model_validate({"diarization": {}})
        assert cfg.diarization.hf_token == "test_token_from_env"

    def test_explicit_hf_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "env_token")
        cfg = PipelineConfig.model_validate({"diarization": {"hf_token": "explicit_token"}})
        assert cfg.diarization.hf_token == "explicit_token"

    def test_invalid_device(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"diarization": {"device": "mps"}})


class TestLLMConfig:
    def test_temperature_bounds(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"llm": {"temperature": -0.1}})
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"llm": {"temperature": 2.1}})

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"llm": {"provider": "openai"}})

    def test_custom_model(self):
        cfg = PipelineConfig.model_validate({"llm": {"model": "llama3.1:8b"}})
        assert cfg.llm.model == "llama3.1:8b"


class TestPipelineSettings:
    def test_valid_output_formats(self):
        for fmt in ("markdown", "plaintext", "json"):
            cfg = PipelineConfig.model_validate({"pipeline": {"output_format": fmt}})
            assert cfg.pipeline.output_format == fmt

    def test_invalid_output_format(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"pipeline": {"output_format": "pdf"}})

    def test_block_duration_bounds(self):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"pipeline": {"block_duration_minutes": 0}})
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"pipeline": {"block_duration_minutes": 61}})


class TestGlossary:
    def test_glossary_empty_list(self):
        cfg = PipelineConfig.model_validate({"glossary": []})
        assert cfg.glossary == []

    def test_glossary_with_terms(self):
        terms = ["FUVEST", "termodinâmica", "eigenvalue"]
        cfg = PipelineConfig.model_validate({"glossary": terms})
        assert cfg.glossary == terms

    def test_glossary_null_becomes_empty(self):
        cfg = PipelineConfig.model_validate({"glossary": None})
        assert cfg.glossary == []
