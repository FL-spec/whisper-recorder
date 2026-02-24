"""
AbstractTranscriber — interface for Piece 1 (Transcription).

The concrete implementation (WhisperTranscriber) lives in modules/transcriber.py.
The orchestrator only ever imports this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.models.word_token import WordToken


class AbstractTranscriber(ABC):
    """
    Transcribes an audio file and returns word-level tokens.

    The implementation MUST:
    - Never cut or modify the audio file.
    - Enable word_timestamps so each token has precise start/end.
    - Use built-in VAD to suppress silence segments.
    - Release all model memory after this method returns (handled by the orchestrator).
    """

    @abstractmethod
    def transcribe(self, audio_path: str, config: PipelineConfig) -> list[WordToken]:
        """
        Args:
            audio_path: Absolute path to the original, unmodified audio file.
            config: Validated pipeline configuration.

        Returns:
            Ordered list of WordToken objects covering the entire recording.
        """
        ...
