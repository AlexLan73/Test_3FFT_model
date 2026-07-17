"""Пространственно-временна́я 3D-БПФ модель на сетке i×j×N (F9: угловые оси
дополняются нулями до 2ⁿ независимо по X и Y, дефолт i=j=16 уже степень двойки)."""
from __future__ import annotations

import numpy as np

from ..config import ArrayConfig, RangeConfig
from .base import RadarModel
from .result import Axis, SpectralCube
from .windows import AxisWindows


class Fft3DModel(RadarModel):
    """3D-БПФ: угловые оси центрируются (boresight=0), дальность односторонняя."""

    def __init__(self, array: ArrayConfig, rng: RangeConfig,
                 windows: AxisWindows | None = None):
        self._array = array
        self._rng = rng
        self._windows = windows or AxisWindows()

    def _apply_windows(self, cube):
        return self._windows.apply(cube)

    def _transform(self, cube):
        pow2x, pow2y = self._array.padded_shape()
        # угловые оси паддятся нулями до 2ⁿ (F9); дальность -- своя длина n_fft
        spectrum = np.fft.fftn(cube, s=(pow2x, pow2y, self._rng.n_fft))
        # центрируем ТОЛЬКО угловые оси; дальность остаётся односторонней (tau>=0)
        return np.fft.fftshift(spectrum, axes=(0, 1))

    def _build_result(self, spectrum):
        pow2x, pow2y = self._array.padded_shape()
        kx = Axis("kx", np.arange(-pow2x // 2, pow2x // 2), centered=True)
        ky = Axis("ky", np.arange(-pow2y // 2, pow2y // 2), centered=True)
        rng = Axis("range", np.arange(self._rng.n_fft), centered=False)
        return SpectralCube(np.abs(spectrum), kx, ky, rng)
