"""Пространственно-временна́я 3D-БПФ модель на сетке 16x16xN."""
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
        nx, ny = self._array.nx, self._array.ny
        spectrum = np.fft.fftn(cube, s=(nx, ny, self._rng.n_fft))
        # центрируем ТОЛЬКО угловые оси; дальность остаётся односторонней (tau>=0)
        return np.fft.fftshift(spectrum, axes=(0, 1))

    def _build_result(self, spectrum):
        nx, ny = self._array.nx, self._array.ny
        kx = Axis("kx", np.arange(-nx // 2, nx // 2), centered=True)
        ky = Axis("ky", np.arange(-ny // 2, ny // 2), centered=True)
        rng = Axis("range", np.arange(self._rng.n_fft), centered=False)
        return SpectralCube(np.abs(spectrum), kx, ky, rng)
