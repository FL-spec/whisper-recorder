"""
AbstractDiarizer — interface for Piece 2 (Speaker Identification).

The concrete implementation (PyannoteDiarizer) lives in modules/diarizer.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.models.speaker_segment import SpeakerSegment


class AbstractDiarizer(ABC):
    """
    Runs speaker diarization on an audio file and returns speaker intervals.

    The implementation MUST:
    - Operate on the same original, unmodified audio file as the transcriber.
    - Never cut or segment the audio physically.
    - Use the same timeline (seconds from file start) as the transcription output.
    - Release all model memory after this method returns (handled by the orchestrator).
    """

    @abstractmethod
    def diarize(self, audio_path: str, config: PipelineConfig) -> list[SpeakerSegment]:
        """
        Args:
            audio_path: Absolute path to the original, unmodified audio file.
            config: Validated pipeline configuration.

        Returns:
            List of SpeakerSegment objects in chronological order.
            May return an empty list if diarization is disabled or fails gracefully.
        """
        ...
