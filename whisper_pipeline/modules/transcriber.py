"""
WhisperTranscriber — Piece 1 (Transcription).

Wraps faster-whisper with word_timestamps=True and built-in Silero VAD.
The audio file is NEVER modified — this module only reads it.

Key difference from v1: word_timestamps=True is mandatory here so that
Piece 3 (TimestampAligner) can assign a speaker to every individual word.
"""
from __future__ import annotations

import logging
from pathlib import Path

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.interfaces.transcriber import AbstractTranscriber
from whisper_pipeline.models.word_token import WordToken

logger = logging.getLogger(__name__)


class WhisperTranscriber(AbstractTranscriber):
    """Concrete transcriber using faster-whisper."""

    def transcribe(self, audio_path: str, config: PipelineConfig) -> list[WordToken]:
        """
        Transcribe audio_path using faster-whisper with word-level timestamps.

        Steps:
        1. Validate the audio file exists.
        2. Load the WhisperModel (downloads from HuggingFace cache if needed).
        3. Transcribe with vad_filter=True and word_timestamps=True.
        4. Flatten all word-level results into a list of WordToken objects.
        5. Return — the orchestrator handles model teardown.

        Args:
            audio_path: Path to the original, unmodified audio file.
            config: Validated pipeline configuration.

        Returns:
            List of WordToken objects in chronological order.
        """
        from faster_whisper import WhisperModel

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

        wc = config.whisper
        logger.info(
            "Loading faster-whisper model '%s' on device='%s', compute_type='%s'…",
            wc.model,
            wc.device,
            wc.compute_type,
        )

        model = WhisperModel(
            wc.model,
            device=wc.device,
            compute_type=wc.compute_type,
        )

        logger.info("Transcribing '%s' (language='%s', beam_size=%d)…", path.name, wc.language, wc.beam_size)

        segments_iter, info = model.transcribe(
            str(path),
            language=wc.language,
            beam_size=wc.beam_size,
            word_timestamps=True,           # CRITICAL — must be True for alignment
            vad_filter=True,                # Skip silence using built-in Silero VAD
            vad_parameters={
                "min_silence_duration_ms": wc.vad_min_silence_ms,
            },
        )

        words: list[WordToken] = []
        segment_count = 0

        for segment in segments_iter:
            segment_count += 1
            if segment.words is None:
                # Shouldn't happen with word_timestamps=True, but be safe
                logger.warning(
                    "Segment [%.2f–%.2f] returned no word-level data; skipping.",
                    segment.start,
                    segment.end,
                )
                continue

            for w in segment.words:
                words.append(
                    WordToken(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                    )
                )

        logger.info(
            "Transcription complete: %d segments → %d word tokens "
            "(detected language: %s, confidence: %.0f%%).",
            segment_count,
            len(words),
            info.language,
            info.language_probability * 100,
        )

        # Store the model reference so the orchestrator can release it
        self._model = model
        return words

    # ── Teardown hook ──────────────────────────────────────────────────────────

    def get_model(self) -> object:
        """Return the loaded model object so the orchestrator can release it."""
        return getattr(self, "_model", None)
