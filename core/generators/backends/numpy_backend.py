"""NumpyBackend — эталонная (CPU/Windows) реализация `GenBackend` (§4.3 спеки, P1).

Тот же приём, что `ThermalNoise` (`core/generators/sources.py:65`) и
`PointSignalGenerator` (`core/snr/signal.py`): `scale=√(power/2)` на I и Q по
отдельности → дисперсия комплексного шума `σ²=power` (R5-математика, IQ-baseband
БЕЗ множителя 2).
"""
from __future__ import annotations

import numpy as np


class NumpyBackend:
    """Чистый numpy — портативно (cp312/cp313), без GPU-зависимостей."""

    def exp_phase(self, phase: np.ndarray) -> np.ndarray:
        return np.exp(1j * phase).astype(np.complex64)

    def apply_window(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # x·mask — bool-маска промотируется в 0/1, результат не мутирует вход.
        return (x * mask).astype(x.dtype)

    def add_noise(self, x: np.ndarray, power: float,
                  rng: np.random.Generator) -> np.ndarray:
        scale = np.sqrt(power / 2.0)
        noise = scale * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
        return (x + noise).astype(x.dtype)
