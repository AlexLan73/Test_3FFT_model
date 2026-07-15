"""Геометрия решётки и параметры дальностной оси (Value Objects)."""
from __future__ import annotations

from dataclasses import dataclass


def _next_pow2(n: int) -> int:
    """Наименьшая степень двойки >= n (n>=1)."""
    p = 1
    while p < n:
        p *= 2
    return p


@dataclass(frozen=True)
class ArrayConfig:
    """Размер приёмной решётки (число элементов по осям).

    Не обязана быть квадратной (F9, SPEC §1): `nx != ny` -- валидная конфигурация
    (например 6x15). `padded_shape()` отдаёт размеры, дополненные нулями до
    ближайшей степени двойки по каждой оси (нужно угловому FFT-фронтенду P6).
    """
    nx: int = 16
    ny: int = 16

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError("Размеры решётки должны быть положительными")

    def padded_shape(self) -> tuple[int, int]:
        """(nx, ny), дополненные нулями до 2ⁿ по каждой оси независимо."""
        return _next_pow2(self.nx), _next_pow2(self.ny)


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
