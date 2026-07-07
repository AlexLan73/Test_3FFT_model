"""Объёмный скаттер куба: точки выше порога, размер/цвет по амплитуде.

Раскладка осей (что куда экрана) параметризована `AxisLayout` -- дефолт
`range_vertical()` воспроизводит прежний вид без регрессии. Выборка точек
(порог, срез дальности, meshgrid) вынесена в общий `CubeSampler`.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..models import SpectralCube
from .layout import AxisLayout
from .sampling import CubeSampler
from .visualizer import Visualizer

# B008: вызов classmethod в дефолте аргумента → module-level синглтон (frozen dataclass, безопасно).
_LAYOUT_VERTICAL: AxisLayout = AxisLayout.range_vertical()


class CubeScatterVisualizer(Visualizer):
    def __init__(self, threshold_db: float = -20.0, range_limit: int | None = None,
                 layout: AxisLayout = _LAYOUT_VERTICAL) -> None:
        self._thr = threshold_db
        self._rmax = range_limit
        self._layout = layout
        self._sampler = CubeSampler(threshold_db=threshold_db, range_limit=range_limit)

    def _limits(self, cube: SpectralCube, axis_key: str) -> tuple[str, tuple[float, float]]:
        """label + limits для оси; для 'range' с активным range_limit -- лимит по срезу,
        не по полной оси куба (иначе рамка графика шире, чем реально нарисованные точки)."""
        _, label, limits = self._layout.resolve(cube, axis_key)
        if axis_key == "range" and self._rmax is not None:
            rvals = cube.range.values[: self._rmax]
            limits = (float(rvals.min()), float(rvals.max()))
        return label, limits

    def render(self, cube: SpectralCube) -> Figure:
        pts = self._sampler.points(cube, self._layout)
        xlabel, xlim = self._limits(cube, self._layout.axis_x)
        ylabel, ylim = self._limits(cube, self._layout.axis_y)
        zlabel, zlim = self._limits(cube, self._layout.axis_z)

        fig = plt.figure(figsize=(8.5, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts.x, pts.y, pts.z,
                   c=pts.values_db, s=(pts.values_db - self._thr) ** 2 * 0.6 + 5, cmap="turbo",
                   vmin=self._thr, vmax=0, alpha=0.6, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_title("3D-БПФ куб: угол центрирован, дальность односторонняя")
        ax.view_init(16, -58)
        fig.tight_layout()
        return fig
