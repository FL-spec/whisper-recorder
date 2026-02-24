"""
PyannoteDiarizer — Piece 2 (Speaker Identification).

Runs pyannote/speaker-diarization-3.1 on the original audio file.
The audio file is NEVER modified — this module only reads it.

The output intervals use exactly the same timeline as faster-whisper,
because both tools read the same original file from second 0, which is
the mathematical foundation that makes Piece 3 (alignment) valid.

Requirements:
- pyannote.audio >= 3.1  (pip install pyannote.audio)
- HuggingFace token with accepted model licence
  (https://hf.co/pyannote/speaker-diarization-3.1)
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.interfaces.diarizer import AbstractDiarizer
from whisper_pipeline.models.speaker_segment import SpeakerSegment

logger = logging.getLogger(__name__)

_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


class PyannoteDiarizer(AbstractDiarizer):
    """Concrete diarizer using pyannote.audio 3.1."""

    def diarize(self, audio_path: str, config: PipelineConfig) -> list[SpeakerSegment]:
        """
        Run speaker diarization on the original audio file.

        If diarization is disabled in config, returns an empty list immediately.
        If pyannote fails for any reason, logs the error and returns an empty
        list so the rest of the pipeline can continue with 'UNKNOWN' speakers.

        Args:
            audio_path: Path to the original, unmodified audio file.
            config: Validated pipeline configuration.

        Returns:
            List of SpeakerSegment objects in chronological order.
        """
        dc = config.diarization

        if not dc.enabled:
            logger.info("Diarization is disabled in config — skipping.")
            self._pipeline = None
            return []

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path.resolve()}")

        if not dc.hf_token:
            logger.warning(
                "No HuggingFace token found (diarization.hf_token / HF_TOKEN env var). "
                "The pyannote model requires a token and an accepted licence. "
                "Skipping diarization — all speakers will be labelled 'UNKNOWN'."
            )
            self._pipeline = None
            return []

        # Import heavy deps here (lazy) so they don't slow down startup
        try:
            # Suppress the torchcodec dylib warning on macOS (missing older libavutil)
            # — it is non-fatal; pyannote uses torchaudio, not torchcodec
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torchcodec.*")
                warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
                from pyannote.audio import Pipeline
                import torch
        except ImportError as exc:
            logger.error(
                "pyannote.audio is not installed. "
                "Install it with: pip install 'pyannote.audio>=3.1'\n%s", exc
            )
            self._pipeline = None
            return []

        logger.info(
            "Loading Pyannote pipeline '%s' on device='%s'…",
            _PYANNOTE_MODEL,
            dc.device,
        )

        try:
            pipeline = Pipeline.from_pretrained(
                _PYANNOTE_MODEL,
                token=dc.hf_token,
            )
        except Exception as exc:
            logger.error(
                "Failed to load Pyannote pipeline. "
                "Make sure you have accepted the model licence at "
                "https://hf.co/%s\nError: %s",
                _PYANNOTE_MODEL,
                exc,
            )
            self._pipeline = None
            return []

        # Move to configured device
        device_obj = torch.device(dc.device)
        pipeline = pipeline.to(device_obj)
        self._pipeline = pipeline

        # Build optional speaker count hints
        diarize_kwargs: dict = {}
        if dc.min_speakers is not None:
            diarize_kwargs["min_speakers"] = dc.min_speakers
        if dc.max_speakers is not None:
            diarize_kwargs["max_speakers"] = dc.max_speakers

        logger.info("Running diarization on '%s'…", path.name)

        try:
            diarization = pipeline(str(path), **diarize_kwargs)
        except Exception as exc:
            logger.error("Diarization failed: %s — continuing with UNKNOWN speakers.", exc)
            return []

        segments: list[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker_id=speaker,
                )
            )

        # Sort by start time (pyannote usually returns sorted, but be explicit)
        segments.sort(key=lambda s: s.start)

        logger.info(
            "Diarization complete: %d segments, %d unique speakers.",
            len(segments),
            len({s.speaker_id for s in segments}),
        )
        return segments

    def get_model(self) -> object:
        """Return the loaded pipeline for the orchestrator to release."""
        return getattr(self, "_pipeline", None)
