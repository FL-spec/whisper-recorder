"""
AbstractSynthesizer — interface for Piece 5 (Reduce Phase / Global Synthesis).

The concrete implementation (OllamaSynthesizer) lives in modules/synthesizer.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.models.block_result import BlockResult
from whisper_pipeline.models.final_output import FinalOutput


class AbstractSynthesizer(ABC):
    """
    Takes all block-level extraction results and synthesises them into a single
    coherent FinalOutput structure via one final LLM call.

    The implementation MUST:
    - Load the synthesis prompt template from an external file.
    - Perform a SINGLE LLM call across all blocks (not one per block).
    - Validate the response with Pydantic before returning.
    - Aggregate logistical notices and student questions from all blocks.
    """

    @abstractmethod
    def synthesize(
        self,
        blocks: list[BlockResult],
        audio_path: str,
        config: PipelineConfig,
    ) -> FinalOutput:
        """
        Args:
            blocks: All BlockResult objects from the extraction phase, in order.
            audio_path: Original audio path (stored in FinalOutput metadata).
            config: Validated pipeline configuration.

        Returns:
            A fully validated FinalOutput object ready to be written to disk.
        """
        ...
