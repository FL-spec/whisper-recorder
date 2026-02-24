"""
AbstractExtractor — interface for Piece 4 (Map Phase / Block Extraction).

The concrete implementation (OllamaExtractor) lives in modules/extractor.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.models.aligned_segment import AlignedSegment
from whisper_pipeline.models.block_result import BlockResult


class AbstractExtractor(ABC):
    """
    Processes one time block of aligned transcript through the LLM and returns
    a structured BlockResult.

    The implementation MUST:
    - Load prompt templates from external files (never inline strings).
    - Perform transcription correction and content extraction in a SINGLE LLM call.
    - Validate the LLM response with Pydantic before returning.
    - Handle LLM JSON parse failures gracefully (retry or return partial result).
    - Include glossary terms in the prompt when the config glossary is non-empty.
    """

    @abstractmethod
    def extract_blocks(
        self,
        segments: list[AlignedSegment],
        config: PipelineConfig,
    ) -> list[BlockResult]:
        """
        Split segments into blocks and run one LLM extraction per block.

        Args:
            segments: Chronologically ordered aligned segments (entire recording).
            config: Validated pipeline configuration (includes block_duration_minutes).

        Returns:
            List of BlockResult objects, one per block, in order.
        """
        ...
