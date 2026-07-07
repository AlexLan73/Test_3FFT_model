"""Дальностные профили выбранных угловых ячеек (наложение)."""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..models import SpectralCube
from .visualizer import Visualizer


class RangeProfileVisualizer(Visualizer):
    """cells: список (kx, ky, подпись) угловых ячеек для сравнения профилей."""

    def __init__(self, cells: Sequence[tuple[float, float, str]],
                 range_limit: int | None = None):
        self._cells = cells
        self._rmax = range_limit

    def render(self, cube: SpectralCube) -> Figure:
        fig, ax = plt.subplots(figsize=(11, 5))
        rmax = self._rmax or cube.range.values.size
        for kx, ky, label in self._cells:
            ix, iy = cube.index_of_angle(kx, ky)
            prof = cube.range_profile_db(ix, iy)[:rmax]
            ax.plot(cube.range.values[:rmax], prof, lw=1.4, label=label)
        ax.set_xlabel("бин дальности (задержка >= 0)")
        ax.set_ylabel("дБ отн. макс ячейки")
        ax.set_ylim(-35, 3)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("Дальностные профили угловых ячеек")
        fig.tight_layout()
        return fig
