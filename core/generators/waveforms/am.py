"""AmWaveform — амплитудная модуляция (Strategy, §5 спеки). Новая формула (нет в DSP-GPU):

    a(t) = (1 + m·cos(2π·f_m·t)) · exp(j·(2π·f0·t + phase))

`m` (индекс модуляции) и `f_m` (частота модуляции, Гц) — в `spec.meta` (G10, как у
любого доп. параметра под конкретный тип волны).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._pipeline import amplitude_for_snr, render_pipeline
from .base import Waveform, WaveformSpec
from .field import Modulation, SignalField

if TYPE_CHECKING:
    from ..backends.base import GenBackend

DEFAULT_M: float = 0.5           # индекс АМ по умолчанию
DEFAULT_F_M_FRACTION: float = 1.0 / 128.0   # f_m = fs/128 по умолчанию (целое число периодов в окне)


def _am_numpy(fs: float, length: int, f0: float, amplitude: float,
              phase: float, m: float, f_m: float) -> np.ndarray:
    """1D АМ-сигнал (шаг 1 пайплайна §4.0). Не мутирует ничего — новый массив."""
    t = np.arange(length, dtype=np.float64) / fs
    envelope = 1.0 + m * np.cos(2.0 * np.pi * f_m * t)
    carrier = np.exp(1j * (2.0 * np.pi * f0 * t + phase))
    return (amplitude * envelope * carrier).astype(np.complex64)


class AmWaveform(Waveform):
    """АМ: несущая `f0` + боковые `f0±f_m` → окно → раскладка n×n → шум (§4.0 пайплайн)."""

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        m = float(spec.meta.get("m", DEFAULT_M))
        f_m = float(spec.meta.get("f_m", spec.fs * DEFAULT_F_M_FRACTION))
        signal = _am_numpy(spec.fs, spec.n_samples, spec.carrier_hz, amplitude, spec.phase, m, f_m)
        return render_pipeline(backend, spec, rng, signal, Modulation.AM)
