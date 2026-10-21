"""Hardware detection and sensible compute defaults.

Kept separate from the engine so it can be tested without loading a model.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

#: compute types that only make sense on a GPU
GPU_ONLY_COMPUTE = {"float16", "int8_float16"}


def cuda_device_count() -> int:
    """Number of usable CUDA devices, or 0.

    Returns 0 rather than raising when CUDA libraries are absent or the driver
    is installed but not working — a very common laptop state.
    """
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception as exc:                       # noqa: BLE001
        log.debug("CUDA probe failed: %s", exc)
        return 0


def default_cpu_threads(requested: int = 0) -> int:
    """Threads for CPU inference. 0 means "let CTranslate2 decide".

    Measured on a 20-core i7-12700H over 120s of audio (whisper-small/int8,
    three interleaved runs each, medians): CTranslate2's own default 8.7s,
    forcing 4 threads 8.9s, forcing 8 threads 10.3s. Oversubscribing the cores
    makes it markedly *slower*, so the library's default is left alone unless
    the user asks for something specific.
    """
    if requested > 0:
        return requested
    env = os.environ.get("OMP_NUM_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return 0


def resolve_device(requested: str, compute_type: str) -> tuple[str, str, list[str]]:
    """Pick the device and a compute type it can actually run.

    Returns (device, compute_type, notes) where notes are messages worth
    showing the user about any substitution made.
    """
    notes: list[str] = []
    gpus = cuda_device_count()

    if requested == "auto":
        device = "cuda" if gpus else "cpu"
        if gpus:
            notes.append(f"Detected {gpus} CUDA device(s); using GPU.")
        else:
            notes.append("No usable CUDA device; using CPU.")
    elif requested == "cuda" and not gpus:
        device = "cpu"
        notes.append("--device cuda requested but no usable CUDA device was found "
                     "(driver missing or not running); falling back to CPU.")
    else:
        device = requested

    if device == "cpu" and compute_type in GPU_ONLY_COMPUTE:
        notes.append(f"compute type {compute_type!r} needs a GPU; using int8 on CPU.")
        compute_type = "int8"

    return device, compute_type, notes
