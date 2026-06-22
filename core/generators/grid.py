"""Геометрия решётки и расчёт фазового вектора наведения."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..config import ArrayConfig


@dataclass(frozen=True)
class ArrayGrid:
    """Сетка элементов решётки. Information Expert по пространственной фазе."""
    nx: int
    ny: int

    @classmethod
    def from_config(cls, cfg: ArrayConfig) -> "ArrayGrid":
        return cls(cfg.nx, cfg.ny)

    def steering(self, kx: float, ky: float) -> np.ndarray:
        """Фазовый вектор наведения (nx, ny) для прихода с углового бина (kx, ky).

        После fftshift по пространственным осям источник проявится в бине (kx, ky).
        """
        ax = np.arange(self.nx)
        ay = np.arange(self.ny)
        phase = 2j * np.pi * (kx * ax[:, None] / self.nx + ky * ay[None, :] / self.ny)
        return np.exp(phase)
