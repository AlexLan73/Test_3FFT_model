"""Запись фигур на диск -- Pure Fabrication (IO отделён от рендера)."""
from __future__ import annotations
import os
from matplotlib.figure import Figure


class FigureWriter:
    def __init__(self, out_dir: str, dpi: int = 120):
        self._dir = out_dir
        self._dpi = dpi
        os.makedirs(out_dir, exist_ok=True)

    def write(self, fig: Figure, name: str) -> str:
        path = os.path.join(self._dir, name)
        fig.savefig(path, dpi=self._dpi)
        return path
