"""Геометрия решётки и параметры дальностной оси (Value Objects)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArrayConfig:
    """Размер квадратной приёмной решётки (число элементов по осям)."""
    nx: int = 16
    ny: int = 16

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("Размеры решётки должны быть положительными")


@dataclass(frozen=True)
class RangeConfig:
    """Дальностная (быстрая) ось.

    n_real -- число реальных отсчётов после дерампа.
    n_fft  -- длина БПФ по дальности; n_fft > n_real означает дополнение нулями
              (интерполяция, односторонняя ось задержки tau >= 0).
    """
    n_real: int = 16
    n_fft: int = 16

    def __post_init__(self) -> None:
        if self.n_fft < self.n_real:
            raise ValueError("n_fft не может быть меньше n_real")

    @property
    def is_zero_padded(self) -> bool:
        return self.n_fft > self.n_real
