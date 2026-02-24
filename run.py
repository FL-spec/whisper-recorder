#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          Whisper Pipeline v2 — Unified Entrypoint                ║
║                                                                  ║
║  Press Enter to start. Press Enter again to stop.                ║
║  The full pipeline runs automatically after recording.           ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python run.py                           # Record then process
    python run.py --max-seconds 120         # Auto-stop after 2 min
    python run.py --no-diarization          # Skip speaker ID (faster)
    python run.py --output-format json      # JSON output
    python run.py --config my_config.yaml   # Custom config
    python run.py --file lecture.wav        # Skip recording, process existing file
    python run.py --list-devices            # Show audio input devices

What happens after you press Enter to stop:
    Step 1 → faster-whisper transcribes audio (word-level timestamps + VAD)
    Step 2 → Pyannote identifies who spoke when
    Step 3 → Python aligns words to speakers
    Step 4 → LLM extracts topics/questions/notices per 10-min block
    Step 5 → LLM synthesises all blocks into one structured summary
    Step 6 → Two files written to outputs/: transcript + summary
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path

# Suppress the torchcodec/libavutil wall of warnings that pyannote.audio emits
# when its optional torchcodec decoder can't find FFmpeg shared libraries.
# This is harmless — pyannote falls back to torchaudio for audio loading.
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
warnings.filterwarnings("ignore", message=".*Could not load libtorchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

from dotenv import load_dotenv
load_dotenv()

# ── Ensure HF token is set for pyannote model download ───────────────────────
_hf = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf:
    os.environ["HF_TOKEN"] = _hf
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf
    try:
        import huggingface_hub as _hh
        _hh.login(token=_hf, add_to_git_credential=False)
    except Exception:
        pass

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

SAMPLE_RATE = 16_000   # Whisper & Pyannote both want 16 kHz
CHANNELS    = 1
DTYPE       = "int16"


# ──────────────────────────────────────────────────────────────────────────────
#  Pre-flight checks
# ──────────────────────────────────────────────────────────────────────────────

def preflight_check(config) -> list[str]:
    """
    Run sanity checks before the recording starts so failures surface
    immediately, not after a 90-minute recording.

    Returns a list of warning strings (empty = all clear).
    """
    warnings = []

    # 1. Check Ollama is reachable
    try:
        import ollama
        ollama.list()  # lightweight ping
    except Exception:
        warnings.append(
            "⚠  Ollama is not running. Start it with:  ollama serve\n"
            "   The LLM extraction steps (4 & 5) will fail without it."
        )

    # 2. Check the LLM model is pulled
    try:
        import ollama
        available = {m["model"] for m in ollama.list().get("models", [])}
        model_tag = config.llm.model
        # Ollama may list model as "llama3.1:8b" or "llama3.1:8b-instruct-q4_0" etc.
        if not any(model_tag in tag for tag in available):
            warnings.append(
                f"⚠  Model '{model_tag}' not found in Ollama.\n"
                f"   Pull it with:  ollama pull {model_tag}"
            )
    except Exception:
        pass  # Ollama already warned above

    # 3. Check pyannote is installed (warn, not fatal — pipeline degrades gracefully)
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        warnings.append(
            "⚠  pyannote.audio is not installed — diarization will be skipped.\n"
            "   Install with:  pip install pyannote.audio>=3.1\n"
            "   All speakers will be labelled 'UNKNOWN' in the transcript."
        )

    # 4. Check HF token if diarization is enabled
    if config.diarization.enabled and not config.diarization.hf_token:
        warnings.append(
            "⚠  No HuggingFace token found (HF_TOKEN in .env).\n"
            "   Diarization requires a token + accepted licence at:\n"
            "   https://hf.co/pyannote/speaker-diarization-3.1"
        )

    return warnings


# ──────────────────────────────────────────────────────────────────────────────
#  Recording
# ──────────────────────────────────────────────────────────────────────────────

def _format_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def record_audio(max_seconds: int | None = None) -> np.ndarray:
    """
    Record from the default microphone until Enter is pressed (or max_seconds).
    Returns a 1-D int16 numpy array at SAMPLE_RATE.
    """
    audio_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            console.print(f"[red]⚠ {status}[/red]")
        audio_queue.put(indata.copy())

    def wait_for_enter():
        input()
        stop_event.set()

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    chunks = []
    start_time = time.time()

    sys.stdout.write("\n\033[1;32m● Gravando…\033[0m  Pressione \033[1mEnter\033[0m para parar\n")
    sys.stdout.flush()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=callback):
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            if max_seconds and elapsed >= max_seconds:
                sys.stdout.write("\n")
                console.print(f"[yellow]⏱  Auto-stop: {max_seconds}s reached.[/yellow]")
                break
            while True:
                try:
                    chunks.append(audio_queue.get_nowait())
                except queue.Empty:
                    break
            sys.stdout.write(f"\r\033[36m{_format_ts(elapsed)}\033[0m  recording…  ")
            sys.stdout.flush()
            time.sleep(0.1)

    sys.stdout.write("\n")
    sys.stdout.flush()

    if not chunks:
        console.print("[red]No audio captured.[/red]")
        sys.exit(1)

    audio_data = np.concatenate(chunks, axis=0).flatten()
    duration = len(audio_data) / SAMPLE_RATE
    console.print(f"[green]✔  Recording done:[/green] {duration:.1f} seconds\n")
    return audio_data


def save_audio(audio: np.ndarray, output_dir: str) -> str:
    """
    Save the recording as a permanent WAV file (not a temp file).

    The file is kept so you can re-run the pipeline on it later:
        python pipeline.py outputs/recording_2026-02-24_20-53.wav

    Returns the absolute path to the saved WAV.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wav_path = Path(output_dir) / f"recording_{timestamp}.wav"
    wavfile.write(str(wav_path), SAMPLE_RATE, audio)
    console.print(f"[dim]📼  Audio saved: {wav_path}[/dim]")
    return str(wav_path)


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Whisper Pipeline v2 — record and process in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--file", "-f",
        metavar="PATH",
        default=None,
        help="Skip recording and process an existing audio file instead.",
    )
    p.add_argument(
        "--config", "-c",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    p.add_argument(
        "--max-seconds", "-s",
        type=int,
        default=None,
        metavar="N",
        help="Auto-stop recording after N seconds.",
    )
    p.add_argument(
        "--no-diarization",
        action="store_true",
        help="Skip speaker diarization. All text will be labelled 'UNKNOWN'.",
    )
    p.add_argument(
        "--output-format",
        choices=["markdown", "plaintext", "json"],
        default=None,
        help="Override the output format from config (markdown/plaintext/json).",
    )
    p.add_argument(
        "--output-dir", "-o",
        default=None,
        metavar="DIR",
        help="Override the output directory from config.",
    )
    p.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate JSON files (words, speakers, aligned segments) for debugging.",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s  —  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if not args.verbose:
        for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub",
                      "pytorch_lightning", "lightning", "pyannote"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    log = logging.getLogger("run")

    # ── Handle --list-devices ──────────────────────────────────────────────────
    if args.list_devices:
        console.print(sd.query_devices())
        sys.exit(0)

    # ── Load and validate config ───────────────────────────────────────────────
    from whisper_pipeline.config import load_config
    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Configuration error: %s", exc)
        sys.exit(1)

    # Apply CLI overrides
    if args.no_diarization:
        config.diarization.enabled = False
    if args.output_format:
        config.pipeline.output_format = args.output_format
    if args.output_dir:
        config.pipeline.output_dir = args.output_dir
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.save_intermediates:
        config.pipeline.save_intermediates = True

    # ── Banner ─────────────────────────────────────────────────────────────────
    console.print(Panel.fit(
        "[bold cyan]🎙  Whisper Pipeline v2[/bold cyan]\n"
        f"[dim]Whisper: {config.whisper.model}  |  "
        f"LLM: {config.llm.model}  |  "
        f"Language: {config.whisper.language.upper()}  |  "
        f"Output: {config.pipeline.output_format}[/dim]",
        border_style="cyan",
    ))

    # ── Pre-flight ─────────────────────────────────────────────────────────────
    warnings = preflight_check(config)
    if warnings:
        console.print()
        for w in warnings:
            console.print(f"[yellow]{w}[/yellow]")
        console.print()

    # ── Step 0: Get the audio file ─────────────────────────────────────────────
    if args.file:
        # Existing file mode — skip recording entirely
        audio_path = str(Path(args.file).resolve())
        if not Path(audio_path).exists():
            log.error("File not found: %s", audio_path)
            sys.exit(1)
        console.print(f"[dim]📂  Using existing file: {audio_path}[/dim]\n")
    else:
        # Live recording mode
        console.print(
            "[bold]Ready to record.[/bold] "
            "Press [bold green]Enter[/bold green] to start…",
            end="",
        )
        input()  # wait for first Enter before recording begins

        audio_data = record_audio(max_seconds=args.max_seconds)
        audio_path = save_audio(audio_data, config.pipeline.output_dir)

    # ── Steps 1–6: Run the full pipeline ──────────────────────────────────────
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

    try:
        orchestrator.run(audio_path, config)
    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
