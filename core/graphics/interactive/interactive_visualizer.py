"""Абстракция интерактивного визуализатора (Strategy, ветка plotly).

Импорт plotly -- ТОЛЬКО внутри пакета `interactive/` (мягкая зависимость,
matplotlib-ветка `core/graphics/*.py` от plotly не зависит, см. rule 06).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import plotly.graph_objects as go

from ...models import SpectralCube


class InteractiveVisualizer(ABC):
    """Превращает SpectralCube в интерактивную plotly-фигуру. Рендер без записи на диск."""

    @abstractmethod
    def render(self, cube: SpectralCube) -> go.Figure:
        ...
