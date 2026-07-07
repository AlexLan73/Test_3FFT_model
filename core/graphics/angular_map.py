"""Угловая карта энергии (интеграл по дальности) с гейтом обзора."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..models import SpectralCube
from .visualizer import Visualizer


class AngularMapVisualizer(Visualizer):
    def __init__(self, gate_kx: float = 0.0, gate_ky: float = 0.0,
                 gate_half: float = 1.5):
        self._gx, self._gy, self._gh = gate_kx, gate_ky, gate_half

    def render(self, cube: SpectralCube) -> Figure:
        e = cube.angular_energy_db()
        kx, ky = cube.kx.values, cube.ky.values
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(e.T, origin="lower", cmap="turbo", vmin=-25, vmax=0,
                       extent=(float(kx[0]), float(kx[-1]) + 1.0, float(ky[0]), float(ky[-1]) + 1.0),
                       aspect="equal")
        ax.add_patch(Rectangle((self._gx - self._gh, self._gy - self._gh),
                               2 * self._gh, 2 * self._gh, fill=False,
                               ec="r", lw=2))
        ax.set_xlabel("kx (азимут)")
        ax.set_ylabel("ky (угол места)")
        ax.set_title("Угловая карта энергии (красный -- гейт обзора)")
        fig.colorbar(im, ax=ax, label="дБ", shrink=0.85)
        fig.tight_layout()
        return fig
