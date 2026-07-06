"""Раскладка осей куба на график -- Value Object (Strategy-параметр).

`Axis` (core/models/result.py) хранит только технические `name/values/centered` --
человекочитаемые подписи и пределы график вычисляет сам, здесь и только здесь.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import SpectralCube

_LABELS: dict[str, str] = {
    "kx": "kx (азимут)",
    "ky": "ky (угол места)",
    "range": "дальность (>= 0)",
}


@dataclass(frozen=True)
class AxisLayout:
    """Куда ложатся оси куба на график. axis_x/y/z ∈ {'kx', 'ky', 'range'}."""

    axis_x: str = "kx"
    axis_y: str = "ky"
    axis_z: str = "range"

    @classmethod
    def range_vertical(cls) -> AxisLayout:
        """Текущий вид: дальность -- вверх (как в CubeScatterVisualizer)."""
        return cls("kx", "ky", "range")

    @classmethod
    def range_in_depth(cls) -> AxisLayout:
        """v2: дальность -- в глубину (Y экрана), угол места (ky) -- вертикаль."""
        return cls("kx", "range", "ky")

    def resolve(self, cube: SpectralCube, axis_key: str) -> tuple[np.ndarray, str, tuple[float, float]]:
        """Отдаёт (values, label, limits) для одной из осей куба по её ключу."""
        axis = getattr(cube, axis_key)
        values = axis.values
        label = _LABELS[axis_key]
        lo, hi = float(values.min()), float(values.max())
        if axis.centered:
            # симметричный паддинг (как set_xlim(-8, 8) в черновиках при данных -8..7)
            pad = 1.0
            lo, hi = lo - pad, hi + pad
        limits = (lo, hi)
        return values, label, limits
