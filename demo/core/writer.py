"""DemoWriter — обёртка над `FigureWriter` для примеров (Pure Fabrication, §3.4 спеки).

Сам не рисует и не решает ЧТО сохранять — только КУДА (`demo/graphics/<example>/`).
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from core.graphics.writer import FigureWriter


class DemoWriter:
    """Пишет фигуры примера в `<root>/<example>/`, создавая каталог по требованию."""

    def __init__(self, example: str, root: Path = Path("demo/graphics")) -> None:
        self._inner = FigureWriter(str(root / example))

    def write(self, fig: Figure, name: str) -> str:
        """Сохранить `fig` под именем `name` (без расширения не гарантируется — как FigureWriter)."""
        return self._inner.write(fig, name)
