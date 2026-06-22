"""Объёмный скаттер куба: точки выше порога, размер/цвет по амплитуде."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .visualizer import Visualizer
from ..models import SpectralCube


class CubeScatterVisualizer(Visualizer):
    def __init__(self, threshold_db: float = -20.0, range_limit: int | None = None):
        self._thr = threshold_db
        self._rmax = range_limit

    def render(self, cube: SpectralCube) -> Figure:
        m = cube.magnitude_db
        rmax = self._rmax or m.shape[2]
        m = m[:, :, :rmax]
        kx, ky = cube.kx.values, cube.ky.values
        rz = cube.range.values[:rmax]
        KX, KY, RZ = np.meshgrid(kx, ky, rz, indexing="ij")
        v = m.ravel()
        sel = v > self._thr
        vv = v[sel]
        fig = plt.figure(figsize=(8.5, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(KX.ravel()[sel], KY.ravel()[sel], RZ.ravel()[sel],
                   c=vv, s=(vv - self._thr) ** 2 * 0.6 + 5, cmap="turbo",
                   vmin=self._thr, vmax=0, alpha=0.6, edgecolors="none")
        ax.set_xlabel("kx (азимут)")
        ax.set_ylabel("ky (угол места)")
        ax.set_zlabel("дальность (задержка >= 0)")
        ax.set_title("3D-БПФ куб: угол центрирован, дальность односторонняя")
        ax.view_init(16, -58)
        fig.tight_layout()
        return fig
