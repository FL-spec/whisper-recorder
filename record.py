#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║         Whisper Recorder  —  Português (PT)              ║
║   Modelo: faster-whisper large-v3-turbo  |  Local Only   ║
╚══════════════════════════════════════════════════════════╝

Usage:
    python record.py                    # interactive (press Enter to stop)
    python record.py --max-seconds 60  # auto-stop after 60s
    python record.py --output my.txt   # custom output file
    python record.py --model large-v3  # use a different model
"""

import argparse
import datetime
import os
import queue
import sys
import tempfile
import threading
import time

from dotenv import load_dotenv
load_dotenv()   # reads .env → sets HF_TOKEN in os.environ

# huggingface_hub checks either of these env vars:
import os as _os
_hf_token = _os.environ.get("HF_TOKEN") or _os.environ.get("HUGGING_FACE_HUB_TOKEN")
if _hf_token:
    # ensure BOTH names are set (different versions of hf_hub look for different ones)
    _os.environ["HF_TOKEN"] = _hf_token
    _os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    try:
        import huggingface_hub as _hh
        _hh.login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        pass   # non-fatal if login call fails

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import print as rprint

# ─── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_MODEL    = "large-v3-turbo"
DEFAULT_LANGUAGE = "pt"            # Portuguese
SAMPLE_RATE      = 16_000          # Whisper expects 16 kHz
CHANNELS         = 1
DTYPE            = "int16"

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def banner():
    console.print(Panel.fit(
        "[bold cyan]🎙  Whisper Recorder[/bold cyan]  —  [yellow]Português (PT)[/yellow]\n"
        f"[dim]Modelo: {DEFAULT_MODEL}  |  100 % local, sem internet[/dim]",
        border_style="cyan",
    ))


def format_timestamp(seconds: float) -> str:
    """Convert seconds to [MM:SS] string."""
    m, s = divmod(int(seconds), 60)
    return f"[{m:02d}:{s:02d}]"


def record_audio(max_seconds: int | None = None) -> np.ndarray:
    """
    Record microphone audio until the user presses Enter (or max_seconds).
    Returns a 1-D int16 numpy array at SAMPLE_RATE.
    """
    audio_queue: queue.Queue = queue.Queue()
    stop_event   = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            console.print(f"[red]⚠  {status}[/red]")
        audio_queue.put(indata.copy())

    # ── listener thread: waits for Enter key ──────────────────────────────────
    def wait_for_enter():
        input()          # blocks until user presses Enter
        stop_event.set()

    listener = threading.Thread(target=wait_for_enter, daemon=True)
    listener.start()

    chunks = []
    start_time = time.time()

    sys.stdout.write("\n\033[1;32m● Gravando…\033[0m  "
                    "Pressione \033[1mEnter\033[0m para parar\n")
    sys.stdout.flush()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    ):
        while not stop_event.is_set():
            elapsed = time.time() - start_time

            # auto-stop if requested
            if max_seconds and elapsed >= max_seconds:
                sys.stdout.write("\n")
                console.print(f"[yellow]⏱  Limite de {max_seconds}s atingido.[/yellow]")
                break

            # drain audio chunks from the queue (non-blocking)
            while True:
                try:
                    chunk = audio_queue.get_nowait()
                    chunks.append(chunk)
                except queue.Empty:
                    break

            # live in-place timer (\r rewrites the same line)
            sys.stdout.write(f"\r\033[36m{format_timestamp(elapsed)}\033[0m  gravando…  ")
            sys.stdout.flush()
            time.sleep(0.1)

    sys.stdout.write("\n")   # close the \r timer line cleanly
    sys.stdout.flush()

    if not chunks:
        console.print("[red]Nenhum áudio capturado.[/red]")
        sys.exit(1)

    audio_data = np.concatenate(chunks, axis=0).flatten()
    duration   = len(audio_data) / SAMPLE_RATE
    console.print(f"[green]✔  Gravação concluída:[/green] {duration:.1f} segundos\n")
    return audio_data


def save_temp_wav(audio: np.ndarray) -> str:
    """Write int16 PCM audio to a temp WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wavfile.write(tmp.name, SAMPLE_RATE, audio)
    return tmp.name


def transcribe(wav_path: str, model_name: str) -> list[dict]:
    """
    Load the faster-whisper model and transcribe the WAV file in Portuguese.
    Returns a list of segment dicts with 'start', 'end', 'text'.
    """
    import os
    from faster_whisper import WhisperModel

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if hf_token:
        console.print(f"[dim]Token HF detectado (✓)[/dim]")
    else:
        console.print("[yellow]⚠  HF_TOKEN não encontrado — download pode ser lento[/yellow]")

    console.print(f"[dim]Carregando modelo [bold]{model_name}[/bold]…[/dim]")
    console.print("[dim](Na primeira execução o modelo será baixado ~800 MB — aguarde, o progresso aparecerá abaixo)[/dim]\n")

    # Apple Silicon: use int8 on cpu — fastest & most memory-efficient
    # Pass the token explicitly so the download is authenticated
    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=None,        # use default HF cache (~/.cache/huggingface)
        local_files_only=False,
    )

    segments_out = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Transcrevendo…", total=None)

        segments, info = model.transcribe(
            wav_path,
            language=DEFAULT_LANGUAGE,
            beam_size=5,
            vad_filter=True,               # skip silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
            word_timestamps=False,
        )

        for seg in segments:
            segments_out.append({
                "start": seg.start,
                "end":   seg.end,
                "text":  seg.text.strip(),
            })
            progress.update(
                task,
                description=f"[cyan]Transcrevendo…[/cyan]  "
                            f"[dim]{format_timestamp(seg.end)}[/dim]  {seg.text.strip()[:60]}",
            )

    detected = info.language
    prob     = info.language_probability
    console.print(
        f"\n[green]✔  Transcrição concluída[/green]  "
        f"[dim](idioma detectado: [bold]{detected}[/bold], "
        f"confiança: {prob:.0%})[/dim]\n"
    )
    return segments_out


def build_output_text(segments: list[dict], include_timestamps: bool) -> str:
    lines = []
    for seg in segments:
        if include_timestamps:
            ts = f"{format_timestamp(seg['start'])} "
        else:
            ts = ""
        lines.append(f"{ts}{seg['text']}")
    return "\n".join(lines)


def save_txt(text: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"# Gravação — {now}\n\n")
        f.write(text)
        f.write("\n\n" + "─" * 60 + "\n\n")
    console.print(f"[bold green]💾  Salvo em:[/bold green] [cyan]{output_path}[/cyan]")


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gravador de voz com Whisper — transcrição em Português",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Modelo faster-whisper a usar (padrão: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output", "-o",
        default="transcricao.txt",
        help="Arquivo de saída .txt (padrão: transcricao.txt)",
    )
    parser.add_argument(
        "--max-seconds", "-s",
        type=int,
        default=None,
        metavar="SEGUNDOS",
        help="Parar automaticamente após N segundos",
    )
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Não incluir marcações de tempo no arquivo de saída",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Listar dispositivos de áudio disponíveis e sair",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    banner()

    if args.list_devices:
        console.print(sd.query_devices())
        sys.exit(0)

    # 1 — Record
    audio = record_audio(max_seconds=args.max_seconds)

    # 2 — Save temp WAV
    wav_path = save_temp_wav(audio)

    try:
        # 3 — Transcribe
        segments = transcribe(wav_path, model_name=args.model)

        if not segments:
            console.print("[yellow]⚠  Nenhum texto detectado na gravação.[/yellow]")
            sys.exit(0)

        # 4 — Print result to terminal
        console.rule("[bold]Transcrição[/bold]")
        full_text = build_output_text(segments, include_timestamps=not args.no_timestamps)
        console.print(Text(full_text))
        console.rule()

        # 5 — Save to file
        save_txt(full_text, args.output)

    finally:
        # always clean up temp file
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    console.print("\n[bold green]✅  Tudo pronto! A fechar…[/bold green]\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
