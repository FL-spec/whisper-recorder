"""
memory.py — Explicit GPU/CPU memory cleanup utilities.

These functions are called by the orchestrator between every heavy stage.
No model is ever left loaded when the next stage begins.

Design decision: the orchestrator is responsible for calling cleanup — the
modules themselves do NOT call it. This keeps cleanup logic centralised and
auditable in one place (orchestrator.py).
"""
from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def release_torch_model(model: object) -> None:
    """
    Move a PyTorch / CTranslate2 model off the GPU and dereference it.

    Works for:
    - faster-whisper WhisperModel (CTranslate2 backend)
    - pyannote Pipeline objects (PyTorch)
    - Any object with a .to() or no special teardown needed

    Args:
        model: The model object to release. Passing None is safe (no-op).
    """
    if model is None:
        return

    # PyTorch models: move to CPU first to free VRAM
    try:
        import torch

        if hasattr(model, "to") and callable(model.to):
            model.to("cpu")
            logger.debug("Moved model to CPU: %s", type(model).__name__)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not move model to CPU (may be CTranslate2): %s", exc)

    # CTranslate2 models expose an explicit unload
    try:
        if hasattr(model, "unload_model") and callable(model.unload_model):
            model.unload_model()
            logger.debug("Called unload_model() on %s", type(model).__name__)
    except Exception as exc:  # noqa: BLE001
        logger.debug("unload_model() failed: %s", exc)

    del model
    logger.debug("Model reference deleted.")


def clear_gpu_cache() -> None:
    """
    Flush the CUDA memory cache (no-op if CUDA is not available or not used).
    Always safe to call.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug("CUDA cache cleared.")
        else:
            logger.debug("CUDA not available — cache clear skipped.")
    except ImportError:
        logger.debug("PyTorch not importable — CUDA cache clear skipped.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error clearing CUDA cache: %s", exc)


def run_gc() -> None:
    """Run Python's garbage collector to free unreferenced objects immediately."""
    collected = gc.collect()
    logger.debug("GC collected %d objects.", collected)


def full_cleanup(model: object = None) -> None:
    """
    Convenience: release a model, clear GPU cache, and run GC in one call.
    This is the standard call between pipeline stages.

    Args:
        model: Optional model to release before GC. Pass None to just GC.
    """
    logger.info("Running full memory cleanup (model=%s)…", type(model).__name__ if model else "None")
    release_torch_model(model)
    clear_gpu_cache()
    run_gc()
    logger.info("Memory cleanup complete.")
