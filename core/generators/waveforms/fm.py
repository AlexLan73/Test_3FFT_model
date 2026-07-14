"""FmInterferenceWaveform — аналоговая ЧМ-помеха (Strategy, §5 спеки, таск P4).

Модель стороннего источника/связного (радиолюбитель вышел на связь), НЕ наш
зонд. Новая формула (нет в DSP-GPU):

    s(t) = amplitude · exp(j·(2π·f0·t + β·sin(2π·f_m·t) + phase))

(интеграл `∫cos(2π·f_m·t)dt = sin(2π·f_m·t)/(2π·f_m)`, поглощено в `β` — индекс
девиации ЧМ). `β` (`beta`) и `f_m` — в `spec.meta` (G10).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._pipeline import amplitude_for_snr, render_pipeline
from .base import Waveform, WaveformSpec
from .field import Modulation, SignalField

if TYPE_CHECKING:
    from ..backends.base import GenBackend

DEFAULT_BETA: float = 2.0                    # индекс ЧМ по умолчанию (умеренная девиация)
DEFAULT_F_M_FRACTION: float = 1.0 / 256.0    # f_m = fs/256 по умолчанию


def _fm_numpy(fs: float, length: int, f0: float, amplitude: float,
              phase: float, beta: float, f_m: float) -> np.ndarray:
    """1D ЧМ-сигнал (шаг 1 пайплайна §4.0). Не мутирует ничего — новый массив."""
    t = np.arange(length, dtype=np.float64) / fs
    inst_phase = 2.0 * np.pi * f0 * t + beta * np.sin(2.0 * np.pi * f_m * t) + phase
    return (amplitude * np.exp(1j * inst_phase)).astype(np.complex64)


class FmInterferenceWaveform(Waveform):
    """ЧМ-помеха: несущая `f0` + синусоидальная девиация фазы → окно → n×n → шум (§4.0)."""

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        beta = float(spec.meta.get("beta", DEFAULT_BETA))
        f_m = float(spec.meta.get("f_m", spec.fs * DEFAULT_F_M_FRACTION))
        signal = _fm_numpy(spec.fs, spec.n_samples, spec.carrier_hz, amplitude, spec.phase, beta, f_m)
        return render_pipeline(backend, spec, rng, signal, Modulation.FM_INTERFERENCE)
