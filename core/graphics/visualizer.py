"""Абстракция визуализатора (Strategy/Polymorphism)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from matplotlib.figure import Figure

from ..models import SpectralCube


class Visualizer(ABC):
    """Превращает SpectralCube в matplotlib-фигуру. Рендер без записи на диск."""

    @abstractmethod
    def render(self, cube: SpectralCube) -> Figure:
        ...
