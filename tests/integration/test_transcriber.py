"""
Integration test — WhisperTranscriber.

Loads the REAL faster-whisper model and runs it on the short fixture audio.

Run only with:
    pytest tests/integration/ -m integration -v

These tests are slow (model download on first run) and require a network
connection for the initial HuggingFace model download.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_AUDIO = Path(__file__).parent.parent / "fixtures" / "short_audio.wav"


@pytest.mark.integration
def test_transcriber_returns_list(tmp_path):
    """WhisperTranscriber should return a list (possibly empty for tone audio)."""
    from whisper_pipeline.config import PipelineConfig
    from whisper_pipeline.modules.transcriber import WhisperTranscriber

    config = PipelineConfig.model_validate({"whisper": {"model": "tiny", "language": "en", "beam_size": 1}})
    transcriber = WhisperTranscriber()
    result = transcriber.transcribe(str(FIXTURE_AUDIO), config)

    assert isinstance(result, list)
    # Each item must be a WordToken with valid timestamps
    for token in result:
        assert hasattr(token, "word")
        assert hasattr(token, "start")
        assert hasattr(token, "end")
        assert token.end >= token.start


@pytest.mark.integration
def test_transcriber_file_not_found():
    from whisper_pipeline.config import PipelineConfig
    from whisper_pipeline.modules.transcriber import WhisperTranscriber

    config = PipelineConfig()
    transcriber = WhisperTranscriber()
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe("/nonexistent/audio.wav", config)
