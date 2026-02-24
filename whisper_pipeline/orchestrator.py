"""
Orchestrator — the single place that enforces pipeline order and memory cleanup.

This is the only module that knows the full execution sequence. Every other
module only knows its own contract. The orchestrator:

1. Calls the transcriber → records its model reference
2. Calls full_cleanup(transcriber_model)             ← GPU/CPU freed
3. Calls the diarizer → records its pipeline reference
4. Calls full_cleanup(diarizer_pipeline)             ← GPU/CPU freed
5. Calls the aligner (pure Python, no cleanup needed)
6. Optionally saves intermediates to disk
7. Calls the extractor (Ollama — runs on separate process, no GPU cleanup needed)
8. Calls the synthesizer
9. Calls the output writer
10. Returns the final output paths

The orchestrator never imports concrete module classes directly — it only
accepts them via constructor injection, ensuring each piece is independently
replaceable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from whisper_pipeline.config import PipelineConfig
from whisper_pipeline.interfaces.aligner import AbstractAligner
from whisper_pipeline.interfaces.diarizer import AbstractDiarizer
from whisper_pipeline.interfaces.extractor import AbstractExtractor
from whisper_pipeline.interfaces.synthesizer import AbstractSynthesizer
from whisper_pipeline.interfaces.transcriber import AbstractTranscriber
from whisper_pipeline.memory import full_cleanup
from whisper_pipeline.models.final_output import FinalOutput
from whisper_pipeline.output_writer import OutputWriter

logger = logging.getLogger(__name__)
console = Console()


class Orchestrator:
    """
    Coordinates the full pipeline from audio → final output.

    All dependencies are injected via the constructor so any concrete
    implementation can be swapped without touching this class.
    """

    def __init__(
        self,
        transcriber: AbstractTranscriber,
        diarizer: AbstractDiarizer,
        aligner: AbstractAligner,
        extractor: AbstractExtractor,
        synthesizer: AbstractSynthesizer,
        output_writer: OutputWriter,
    ) -> None:
        self._transcriber = transcriber
        self._diarizer = diarizer
        self._aligner = aligner
        self._extractor = extractor
        self._synthesizer = synthesizer
        self._output_writer = output_writer

    def run(self, audio_path: str, config: PipelineConfig) -> FinalOutput:
        """
        Execute the complete pipeline on the given audio file.

        Args:
            audio_path: Absolute path to the original, unmodified audio file.
            config: Validated pipeline configuration.

        Returns:
            FinalOutput with transcript_path and summary_path populated.
        """
        audio_path = str(Path(audio_path).resolve())
        console.print(Panel.fit(
            f"[bold cyan]🎙  Whisper Pipeline v2[/bold cyan]\n"
            f"[dim]Audio: {Path(audio_path).name}[/dim]",
            border_style="cyan",
        ))

        # ── Piece 1: Transcription ─────────────────────────────────────────────
        console.rule("[bold]Step 1 — Transcription[/bold]")
        console.print(
            f"[dim]Model: {config.whisper.model}  |  "
            f"Language: {config.whisper.language.upper()}  |  "
            f"Device: {config.whisper.device}[/dim]"
        )

        with self._spinner("Transcribing audio with word-level timestamps…"):
            word_tokens = self._transcriber.transcribe(audio_path, config)

        console.print(f"[green]✔  Transcription complete:[/green] {len(word_tokens)} word tokens")

        whisper_model = getattr(self._transcriber, "get_model", lambda: None)()
        self._cleanup_stage("transcription", whisper_model)

        # ── Piece 2: Diarization ───────────────────────────────────────────────
        console.rule("[bold]Step 2 — Speaker Diarization[/bold]")
        if config.diarization.enabled:
            console.print(f"[dim]Device: {config.diarization.device}[/dim]")
        else:
            console.print("[yellow]⚠  Diarization disabled — all speakers will be 'UNKNOWN'[/yellow]")

        with self._spinner("Running speaker diarization…"):
            speaker_segments = self._diarizer.diarize(audio_path, config)

        unique_speakers = len({s.speaker_id for s in speaker_segments})
        console.print(
            f"[green]✔  Diarization complete:[/green] "
            f"{len(speaker_segments)} intervals, {unique_speakers} speakers"
        )

        diarizer_model = getattr(self._diarizer, "get_model", lambda: None)()
        self._cleanup_stage("diarization", diarizer_model)

        # ── Piece 3: Alignment ─────────────────────────────────────────────────
        console.rule("[bold]Step 3 — Timestamp Alignment[/bold]")

        with self._spinner("Aligning words to speakers…"):
            aligned_segments = self._aligner.align(word_tokens, speaker_segments, config)

        console.print(f"[green]✔  Alignment complete:[/green] {len(aligned_segments)} segments")

        # Optionally save intermediates
        if config.pipeline.save_intermediates:
            self._save_intermediates(
                audio_path, word_tokens, speaker_segments, aligned_segments, config
            )

        # ── Piece 4: Block Extraction (Map) ────────────────────────────────────
        console.rule("[bold]Step 4 — Block Extraction (Map Phase)[/bold]")
        console.print(
            f"[dim]Model: {config.llm.model}  |  "
            f"Block size: {config.pipeline.block_duration_minutes} min  |  "
            f"Ollama: {config.llm.base_url}[/dim]"
        )

        with self._spinner("Extracting content from blocks…"):
            block_results = self._extractor.extract_blocks(aligned_segments, config)

        console.print(f"[green]✔  Extraction complete:[/green] {len(block_results)} blocks processed")

        # ── Piece 5: Global Synthesis (Reduce) ─────────────────────────────────
        console.rule("[bold]Step 5 — Global Synthesis (Reduce Phase)[/bold]")

        with self._spinner("Synthesising all blocks into global structure…"):
            final_output = self._synthesizer.synthesize(block_results, audio_path, config)

        console.print("[green]✔  Synthesis complete[/green]")

        # ── Output Writing ──────────────────────────────────────────────────────
        console.rule("[bold]Step 6 — Writing Output[/bold]")
        final_output = self._output_writer.write(
            final_output, aligned_segments, audio_path, config
        )

        # ── Done ────────────────────────────────────────────────────────────────
        console.print()
        console.print(Panel.fit(
            f"[bold green]✅  Pipeline complete![/bold green]\n\n"
            f"[cyan]📄 Transcript:[/cyan]  {final_output.transcript_path}\n"
            f"[cyan]📊 Summary:  [/cyan]  {final_output.summary_path}",
            border_style="green",
        ))

        return final_output

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _cleanup_stage(stage_name: str, model: object) -> None:
        logger.info("Cleaning up memory after %s stage…", stage_name)
        full_cleanup(model)
        logger.info("%s memory released.", stage_name.capitalize())

    @staticmethod
    def _spinner(message: str):
        """Context manager that shows a Rich spinner while work is running."""
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]{message}[/cyan]"),
            transient=True,
            console=console,
        )

    @staticmethod
    def _save_intermediates(
        audio_path: str,
        word_tokens,
        speaker_segments,
        aligned_segments,
        config: PipelineConfig,
    ) -> None:
        """Save intermediate JSON files when save_intermediates=true in config."""
        out_dir = Path(config.pipeline.output_dir)
        stem = Path(audio_path).stem

        paths = {
            f"{stem}_words.json": [w.model_dump() for w in word_tokens],
            f"{stem}_speakers.json": [s.model_dump() for s in speaker_segments],
            f"{stem}_aligned.json": [a.model_dump() for a in aligned_segments],
        }

        for filename, data in paths.items():
            p = out_dir / filename
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Saved intermediate: %s", p)
