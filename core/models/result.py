"""Результат обработки -- неизменяемый Value Object со своей семантикой осей."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Axis:
    name: str
    values: np.ndarray
    centered: bool          # True -> симметрична вокруг нуля (угол); False -> односторонняя (дальность)


class SpectralCube:
    """Спектральный куб |C| с метаданными осей. Information Expert по выборкам."""

    def __init__(self, magnitude: np.ndarray, kx: Axis, ky: Axis, rng: Axis):
        self._mag = magnitude
        self.kx, self.ky, self.range = kx, ky, rng

    @property
    def magnitude(self) -> np.ndarray:
        return self._mag

    @property
    def magnitude_db(self) -> np.ndarray:
        m = 20.0 * np.log10(self._mag + 1e-12)
        return m - m.max()

    def index_of_angle(self, kx: float, ky: float) -> tuple[int, int]:
        ix = int(np.argmin(np.abs(self.kx.values - kx)))
        iy = int(np.argmin(np.abs(self.ky.values - ky)))
        return ix, iy

    def angular_energy_db(self) -> np.ndarray:
        """Энергия, проинтегрированная по дальности (nx, ny), в дБ отн. макс."""
        e = np.sqrt((self._mag ** 2).sum(axis=2))
        e_db = 20.0 * np.log10(e + 1e-12)
        return e_db - e_db.max()

    def range_profile_db(self, ix: int, iy: int) -> np.ndarray:
        prof = 20.0 * np.log10(self._mag[ix, iy, :] + 1e-12)
        return prof - prof.max()
