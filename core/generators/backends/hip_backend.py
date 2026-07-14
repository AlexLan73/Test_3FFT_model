"""HipBackend — боевой GPU-бэкенд (§4.3 спеки, P2).

Обвязка над `dsp_signal_generators.FormSignalGeneratorROCm` (DSP-GPU, cp313,
копия `.so` в `core/gpu_libs/`, R1 §2.2.1). Тот же контракт `GenBackend` (LSP) —
`exp_phase`/`apply_window`/`add_noise` идентичны `NumpyBackend` (эти примитивы
элементные, GPU не даёт выигрыша на этом масштабе; боевая часть P2 — сырой
несущий сигнал шага 1 пайплайна (§4.0), который считает GPU-ядро, а не
`reference.cw_numpy`/`getX_numpy`).

`norm=1.0` передаётся явно в `set_params` (решение Alex 2026-07-14, эскалация
P2/G11 п.3) — GPU-дефолт `norm=1/√2` иначе даёт амплитуду ×0.7071 против эталона.

Не импортируется из `core/generators/backends/__init__.py` (опциональный,
GPU-only модуль) — используется явным `from core.generators.backends.hip_backend
import HipBackend`, обёрнутым в try/except → `SkipTest` в тестах/демо, если
`.so`/ROCm недоступны (Windows, cp312, нет устройства).
"""
from __future__ import annotations

import numpy as np

from common.gpu_context import GPUContextManager
from core.gpu_libs import loader as gpu_libs
from core.gpu_libs.loader import GpuLibsUnavailableError

from ..waveforms._pipeline import amplitude_for_snr, render_pipeline
from ..waveforms.base import WaveformSpec
from ..waveforms.field import Modulation, SignalField

_SUPPORTED = (Modulation.CW, Modulation.LFM)


class HipBackend:
    """GenBackend-совместимый (LSP) боевой бэкенд: сырую несущую считает GPU.

    Raises:
        GpuLibsUnavailableError: при создании, если `.so`/ROCm недоступны — тесты/демо
        ловят это и делают `SkipTest`/фолбэк на `NumpyBackend`.
    """

    def __init__(self) -> None:
        gpu_libs.require()
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise GpuLibsUnavailableError("ROCmGPUContext недоступен (ROCm-девайс не создан).")
        self._ctx = ctx
        self._sg = gpu_libs.load("dsp_signal_generators")

    # ── GenBackend Protocol (§4.3) — те же примитивы, что NumpyBackend ──────
    def exp_phase(self, phase: np.ndarray) -> np.ndarray:
        return np.exp(1j * phase).astype(np.complex64)

    def apply_window(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return (x * mask).astype(x.dtype)

    def add_noise(self, x: np.ndarray, power: float,
                  rng: np.random.Generator) -> np.ndarray:
        scale = np.sqrt(power / 2.0)
        noise = scale * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
        return (x + noise).astype(x.dtype)

    # ── Боевой синтез несущей (шаг 1 пайплайна §4.0) на GPU ─────────────────
    def _generate_raw(self, spec: WaveformSpec, fdev_hz: float) -> np.ndarray:
        """Один антенный канал — GPU-эквивалент `reference.cw_numpy`/`getX_numpy`."""
        amplitude = amplitude_for_snr(spec)
        gen = self._sg.FormSignalGeneratorROCm(self._ctx)
        gen.set_params(
            antennas=1, points=spec.n_samples, fs=spec.fs, f0=spec.carrier_hz,
            amplitude=amplitude, phase=spec.phase, fdev=fdev_hz,
            norm=1.0,                        # G11 п.3 решения Alex — не GPU-дефолт 1/√2
            noise_amplitude=0.0, noise_seed=0,
            tau_base=spec.tau_s, tau_step=0.0, tau_min=0.0, tau_max=0.0, tau_seed=12345,
        )
        return np.asarray(gen.generate()[0], dtype=np.complex64)

    def render(self, modulation: Modulation, spec: WaveformSpec,
               rng: np.random.Generator) -> SignalField:
        """Аналог `Waveform.render`, но шаг 1 (несущая) считает GPU. CW/ЛЧМ (§5)."""
        if modulation not in _SUPPORTED:
            raise ValueError(
                f"HipBackend.render: модуляция {modulation!r} не поддержана "
                f"GPU-генератором (доступны: {_SUPPORTED})"
            )
        fdev_hz = spec.fdev_hz if modulation is Modulation.LFM else 0.0
        raw = self._generate_raw(spec, fdev_hz)
        return render_pipeline(self, spec, rng, raw, modulation)
