"""Оконные функции (Strategy) и их применение по трём осям куба."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class WindowFunction(ABC):
    """Стратегия весового окна по одной оси."""

    @abstractmethod
    def taper(self, n: int) -> np.ndarray:
        ...


class RectWindow(WindowFunction):
    def taper(self, n):
        return np.ones(n)


class HannWindow(WindowFunction):
    def taper(self, n):
        return np.hanning(n)


class HammingWindow(WindowFunction):
    def taper(self, n):
        return np.hamming(n)


class AxisWindows:
    """Тройка окон (по двум угловым осям и по дальности).

    Чебышёв/Тейлор для подавления угловых боковиков подключаются добавлением
    новой WindowFunction -- без изменения этого класса (Open/Closed).
    """

    def __init__(self, x: WindowFunction | None = None,
                 y: WindowFunction | None = None,
                 t: WindowFunction | None = None):
        self._x = x or HannWindow()
        self._y = y or HannWindow()
        self._t = t or HannWindow()

    def apply(self, cube: np.ndarray) -> np.ndarray:
        nx, ny, nt = cube.shape
        wx = self._x.taper(nx)[:, None, None]
        wy = self._y.taper(ny)[None, :, None]
        wt = self._t.taper(nt)[None, None, :]
        return cube * wx * wy * wt
