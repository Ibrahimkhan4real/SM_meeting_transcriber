"""Whisper model loading and transcription."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .hardware import default_cpu_threads, resolve_device

log = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    segments: list
    duration: float
    language: str


class Transcriber:
    """Loads the Whisper model once and transcribes many files with it."""

    def __init__(self, model_name: str, device: str = "cpu",
                 compute_type: str = "int8", batch_size: int = 0,
                 cpu_threads: int = 0):
        from faster_whisper import WhisperModel

        device, compute_type, notes = resolve_device(device, compute_type)
        for note in notes:
            log.info("%s", note)

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.cpu_threads = default_cpu_threads(cpu_threads) if device == "cpu" else 0

        detail = f"{device}/{compute_type}"
        if device == "cpu" and self.cpu_threads:
            detail += f", {self.cpu_threads} threads"
        log.info("Loading '%s' model (%s)...", model_name, detail)

        kwargs = {"device": device, "compute_type": compute_type}
        if device == "cpu" and self.cpu_threads:
            kwargs["cpu_threads"] = self.cpu_threads
        try:
            self.model = WhisperModel(model_name, **kwargs)
        except Exception as exc:                    # noqa: BLE001
            if device == "cpu":
                raise
            # A GPU that probes fine can still fail to initialise (out of
            # memory, driver mismatch). Falling back beats failing the run.
            log.warning("GPU initialisation failed (%s); retrying on CPU.", exc)
            self.device, self.compute_type = "cpu", "int8"
            self.cpu_threads = default_cpu_threads(cpu_threads)
            fallback = {"device": "cpu", "compute_type": "int8"}
            if self.cpu_threads:
                fallback["cpu_threads"] = self.cpu_threads
            self.model = WhisperModel(model_name, **fallback)

        self.pipeline = None
        if batch_size > 1:
            from faster_whisper import BatchedInferencePipeline
            self.pipeline = BatchedInferencePipeline(model=self.model)
            log.info("Batched inference enabled (batch size %d).", batch_size)

    def transcribe(self, path: Path, *, language: str = "en", vad: bool = True,
                   word_timestamps: bool = True, progress=None,
                   on_duration=None) -> TranscriptionResult:
        kwargs = dict(
            language=None if language == "auto" else language,
            condition_on_previous_text=False,
            beam_size=5,
            vad_filter=vad,
            word_timestamps=word_timestamps,
        )
        if self.pipeline is not None:
            segments, info = self.pipeline.transcribe(str(path), batch_size=self.batch_size,
                                                      **kwargs)
        else:
            segments, info = self.model.transcribe(str(path), **kwargs)

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if on_duration is not None and duration > 0:
            on_duration(duration)
        collected = []
        previous_end = 0.0
        for seg in segments:
            collected.append(seg)
            if progress is not None:
                progress(max(0.0, seg.end - previous_end))
            previous_end = seg.end

        return TranscriptionResult(
            segments=collected,
            duration=duration,
            language=getattr(info, "language", language) or language,
        )
