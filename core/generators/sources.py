"""Абстракция источника сигнала и базовые источники (Strategy + Composite leaf)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..config import RangeConfig
from .grid import ArrayGrid


class SignalSource(ABC):
    """Любой вклад в куб данных. Источник сам знает, как себя синтезировать.

    Возвращает массив (nx, ny, n_real) комплексных отсчётов быстрого времени.
    """

    @abstractmethod
    def contribute(self, grid: ArrayGrid, rng: RangeConfig,
                   rs: np.random.Generator) -> np.ndarray:
        ...

    @staticmethod
    def _empty(grid: ArrayGrid, rng: RangeConfig) -> np.ndarray:
        return np.zeros((grid.nx, grid.ny, rng.n_real), dtype=complex)


class _SteeredTone(SignalSource):
    """Вспомогательный базовый: дерампнутый тон под заданным углом."""

    def __init__(self, kx: float, ky: float):
        self._kx, self._ky = kx, ky

    def _tone(self, range_bin: float, amplitude: float, phase: float,
              rng: RangeConfig) -> np.ndarray:
        k = np.arange(rng.n_real)
        freq = range_bin / rng.n_fft           # бин дальности -> норм. частота биений
        return amplitude * np.exp(1j * (2 * np.pi * freq * k + phase))

    def _steer(self, grid: ArrayGrid) -> np.ndarray:
        return grid.steering(self._kx, self._ky)


class PointTarget(_SteeredTone):
    """Истинная точечная цель: один пик на дальности range_bin."""

    def __init__(self, kx: float, ky: float, range_bin: float,
                 amplitude: float = 1.0, phase: float = 0.0):
        super().__init__(kx, ky)
        self._range_bin = range_bin
        self._amp = amplitude
        self._phase = phase

    def contribute(self, grid, rng, rs):
        tone = self._tone(self._range_bin, self._amp, self._phase, rng)
        return self._steer(grid)[:, :, None] * tone[None, None, :]


class ThermalNoise(SignalSource):
    """Тепловой шум приёмника: независим по элементам, без направления."""

    def __init__(self, power: float):
        self._power = power

    def contribute(self, grid, rng, rs):
        shape = (grid.nx, grid.ny, rng.n_real)
        scale = np.sqrt(self._power / 2.0)
        return scale * (rs.standard_normal(shape) + 1j * rs.standard_normal(shape))
