"""
AbstractAligner — interface for Piece 3 (Timestamp Alignment).

The concrete implementation (TimestampAligner) lives in modules/aligner.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.models.aligned_segment import AlignedSegment
from whisper_pipeline.models.speaker_segment import SpeakerSegment
from whisper_pipeline.models.word_token import WordToken


class AbstractAligner(ABC):
    """
    Merges word-level transcription tokens with speaker intervals to produce
    aligned segments where each block of text knows who said it and when.

    The implementation MUST:
    - Be pure Python (no ML models, no GPU).
    - Handle the case where some words fall outside all speaker intervals (label as 'UNKNOWN').
    - Produce segments in strict chronological order.
    - Be fully testable in isolation without loading any heavy models.
    """

    @abstractmethod
    def align(
        self,
        words: list[WordToken],
        speakers: list[SpeakerSegment],
        config: PipelineConfig,
    ) -> list[AlignedSegment]:
        """
        Args:
            words: Ordered word tokens from the transcriber.
            speakers: Ordered speaker intervals from the diarizer.
            config: Validated pipeline configuration.

        Returns:
            Ordered list of AlignedSegment objects.
        """
        ...
