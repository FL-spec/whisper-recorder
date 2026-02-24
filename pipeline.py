#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║          Whisper Pipeline v2 — Post-Processing Entrypoint    ║
║  Transcription · Diarization · Alignment · LLM Extraction   ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python pipeline.py recording.wav
    python pipeline.py recording.m4a --config my_config.yaml
    python pipeline.py recording.wav --no-diarization
    python pipeline.py recording.wav --output-format json
    python pipeline.py recording.wav --save-intermediates

After the recording ends, the entire pipeline runs automatically:
    1. faster-whisper   → word-level transcript (with VAD)
    2. Pyannote 3.1     → speaker intervals
    3. Pure Python      → who said what (timestamp alignment)
    4. Ollama (map)     → block-level extraction (correct + extract in one pass)
    5. Ollama (reduce)  → global synthesis
    6. OutputWriter     → summary + transcript written to disk
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load .env early so HF_TOKEN etc. are available


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s  —  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Suppress overly verbose third-party loggers unless in verbose mode
    if not verbose:
        for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whisper Pipeline v2 — transcribe, diarise, and summarise lecture recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to process (.wav, .mp3, .m4a, .flac, …)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Skip speaker diarization (all text labelled 'UNKNOWN')",
    )
    parser.add_argument(
        "--output-format", "-f",
        choices=["markdown", "plaintext", "json"],
        default=None,
        help="Override config output_format for this run",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        metavar="DIR",
        help="Override config output_dir for this run",
    )
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate JSON files (words, speaker segments, aligned segments)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_logging(args.verbose)
    log = logging.getLogger("pipeline")

    # ── 1. Validate audio file ─────────────────────────────────────────────────
    audio_path = Path(args.audio_file).resolve()
    if not audio_path.exists():
        log.error("Audio file not found: %s", audio_path)
        sys.exit(1)

    # ── 2. Load and validate config ────────────────────────────────────────────
    from whisper_pipeline.config import load_config

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Configuration error: %s", exc)
        sys.exit(1)

    # ── 3. Apply CLI overrides ─────────────────────────────────────────────────
    if args.no_diarization:
        config.diarization.enabled = False
        log.info("Diarization disabled via --no-diarization flag.")

    if args.output_format:
        config.pipeline.output_format = args.output_format

    if args.output_dir:
        config.pipeline.output_dir = args.output_dir
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.save_intermediates:
        config.pipeline.save_intermediates = True

    # ── 4. Wire up concrete implementations ───────────────────────────────────
    from whisper_pipeline.modules.aligner import TimestampAligner
    from whisper_pipeline.modules.diarizer import PyannoteDiarizer
    from whisper_pipeline.modules.extractor import OllamaExtractor
    from whisper_pipeline.modules.synthesizer import OllamaSynthesizer
    from whisper_pipeline.modules.transcriber import WhisperTranscriber
    from whisper_pipeline.orchestrator import Orchestrator
    from whisper_pipeline.output_writer import OutputWriter

    orchestrator = Orchestrator(
        transcriber=WhisperTranscriber(),
        diarizer=PyannoteDiarizer(),
        aligner=TimestampAligner(),
        extractor=OllamaExtractor(),
        synthesizer=OllamaSynthesizer(),
        output_writer=OutputWriter(),
    )

    # ── 5. Run the pipeline ────────────────────────────────────────────────────
    try:
        orchestrator.run(str(audio_path), config)
    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
