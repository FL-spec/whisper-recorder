"""
Integration test — PyannoteDiarizer.

Loads the REAL Pyannote 3.1 pipeline. Requires:
- HF_TOKEN in .env with the pyannote model licence accepted.
- pyannote.audio >= 3.1 installed.

Run only with:
    pytest tests/integration/ -m integration -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_AUDIO = Path(__file__).parent.parent / "fixtures" / "short_audio.wav"


@pytest.mark.integration
def test_diarizer_returns_list_or_empty():
    """
    PyannoteDiarizer must return a list. For 5s of pure tone, it may return
    zero or one segment — either is valid. What matters is the type contract.
    """
    from whisper_pipeline.config import PipelineConfig
    from whisper_pipeline.modules.diarizer import PyannoteDiarizer

    config = PipelineConfig()
    diarizer = PyannoteDiarizer()
    result = diarizer.diarize(str(FIXTURE_AUDIO), config)

    assert isinstance(result, list)
    for seg in result:
        assert hasattr(seg, "start")
        assert hasattr(seg, "end")
        assert hasattr(seg, "speaker_id")
        assert seg.end >= seg.start


@pytest.mark.integration
def test_diarizer_disabled_returns_empty():
    from whisper_pipeline.config import PipelineConfig
    from whisper_pipeline.modules.diarizer import PyannoteDiarizer

    config = PipelineConfig.model_validate({"diarization": {"enabled": False}})
    diarizer = PyannoteDiarizer()
    result = diarizer.diarize(str(FIXTURE_AUDIO), config)
    assert result == []
