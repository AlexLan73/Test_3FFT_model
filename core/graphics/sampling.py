"""Порог + выборка точек куба -- общий код обеих веток (matplotlib/plotly).

Pure Fabrication: не является частью домена (SpectralCube), но нужен обеим
стратегиям рендера, поэтому вынесен в отдельный класс, а не дублируется.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import SpectralCube
from .layout import AxisLayout


@dataclass(frozen=True)
class SampledPoints:
    """Точки куба выше порога, разложенные по осям экрана согласно AxisLayout."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    values_db: np.ndarray
    mask: np.ndarray


class CubeSampler:
    """Выбирает точки куба выше `threshold_db`, режет дальность до `range_limit`."""

    def __init__(self, threshold_db: float, range_limit: int | None = None) -> None:
        self._threshold = threshold_db
        self._range_limit = range_limit

    def points(self, cube: SpectralCube, layout: AxisLayout) -> SampledPoints:
        # 1. нормировка -- ГЛОБАЛЬНЫЙ max всего куба (Information Expert), не пересчитывать после обрезки.
        m = cube.magnitude_db

        # 2. срез range_limit по дальностной оси куба (axis=2), независимо от раскладки экрана.
        rmax = self._range_limit or m.shape[2]
        m = m[:, :, :rmax]

        axes_values = {
            "kx": cube.kx.values,
            "ky": cube.ky.values,
            "range": cube.range.values[:rmax],
        }
        # порядок осей самого куба фиксирован (kx, ky, range) -- meshgrid строим в этом же
        # порядке, а затем раскладываем по экранным осям через layout.
        cx, cy, cz = np.meshgrid(axes_values["kx"], axes_values["ky"], axes_values["range"],
                                 indexing="ij")
        cube_axes = {"kx": cx, "ky": cy, "range": cz}

        values_db = m.ravel()
        mask = values_db > self._threshold

        x = cube_axes[layout.axis_x].ravel()[mask]
        y = cube_axes[layout.axis_y].ravel()[mask]
        z = cube_axes[layout.axis_z].ravel()[mask]
        values_db = values_db[mask]

        return SampledPoints(x=x, y=y, z=z, values_db=values_db, mask=mask)
