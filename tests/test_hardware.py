"""Device resolution — must degrade gracefully, never crash the run."""

import pytest

from transcribe import hardware
from transcribe.hardware import default_cpu_threads, resolve_device


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "cuda_device_count", lambda: 0)


@pytest.fixture
def one_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "cuda_device_count", lambda: 1)


def test_auto_picks_cpu_without_gpu(no_gpu):
    device, compute, notes = resolve_device("auto", "int8")
    assert device == "cpu" and compute == "int8"
    assert any("CPU" in n for n in notes)


def test_auto_picks_gpu_when_present(one_gpu):
    device, _, notes = resolve_device("auto", "float16")
    assert device == "cuda"
    assert any("CUDA" in n for n in notes)


def test_explicit_cuda_falls_back_when_driver_is_dead(no_gpu):
    """The user's exact situation: a T600 present but nvidia-smi failing."""
    device, compute, notes = resolve_device("cuda", "float16")
    assert device == "cpu", "must not crash when CUDA is unusable"
    assert compute == "int8", "float16 is not usable on CPU"
    assert any("falling back to CPU" in n for n in notes)


def test_gpu_only_compute_type_downgraded_on_cpu(no_gpu):
    _, compute, notes = resolve_device("cpu", "int8_float16")
    assert compute == "int8"
    assert any("needs a GPU" in n for n in notes)


def test_explicit_cpu_int8_is_left_alone(no_gpu):
    assert resolve_device("cpu", "int8") == ("cpu", "int8", [])


def test_cuda_probe_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("libcudart missing")
    monkeypatch.setattr("ctranslate2.get_cuda_device_count", boom)
    assert hardware.cuda_device_count() == 0


def test_cpu_threads_respects_explicit_request():
    assert default_cpu_threads(12) == 12


def test_cpu_threads_auto_defers_to_the_library(monkeypatch):
    """Forcing a thread count measured slower than CTranslate2's own default."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert default_cpu_threads(0) == 0


def test_cpu_threads_honours_omp_env(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    assert default_cpu_threads(0) == 6
