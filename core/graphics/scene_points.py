"""Общая база рисования 3D-сцен ТОЧЕК — для ВСЕХ потребителей (Strategy + VO, правило 06).

«Любой новый график = подкласс стратегии, а не отдельный скрипт»: сцена точек
(пики сканов, детекции, объекты) рисуется ОДНИМ визуализатором
`ScenePointsVisualizer` везде (core-демки, demo-серия ex2+), а не fig_*-функциями
в каждом. Раскладка осей — реюз `AxisLayout` (VO): дефолт `range_in_depth()` —
**дальность/позиция лежит ПО ГОРИЗОНТУ** (в глубину экрана), ky — вертикаль,
как у остальной графики куба.

Отличие от `CubeScatterVisualizer`: тот рисует ОДИН `SpectralCube` (сплошной
скаттер вокселей), этот — разреженный список точек-детекций произвольного
происхождения (пики многих окон, сцена целиком). Контракт тот же (Strategy):
`render(...) -> Figure`, на диск НЕ пишет (запись — дело writer'а в Composition Root).
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .layout import AxisLayout  # noqa: E402

# B008: classmethod в дефолте → module-level синглтон (как в cube_scatter.py).
_LAYOUT_DEPTH: AxisLayout = AxisLayout.range_in_depth()

_LABELS: dict[str, str] = {
    "kx": "kx (азимут, бины)",
    "ky": "ky (угол места, бины)",
    "range": "позиция по оси (отсчёт)",
}


@dataclass(frozen=True)
class ScenePoint:
    """Точка сцены (VO): пик скана в координатах (kx, ky, позиция) + уровень дБ."""

    kx: float
    ky: float
    range_pos: float
    db: float


@dataclass(frozen=True)
class SceneMarker:
    """Маркер истины (VO): где объект на самом деле (для сверки глазами)."""

    kx: float
    ky: float
    range_pos: float
    label: str


class ScenePointsVisualizer:
    """Strategy: 3D-сцена точек. Раскладка осей — `AxisLayout` (дефолт: дальность по горизонту).

    Один класс на все сцены demo-серии — НЕ перерисовываем в каждом примере заново.
    """

    def __init__(self, layout: AxisLayout = _LAYOUT_DEPTH, vmin_db: float = -25.0,
                 vmax_db: float = 0.0, view: tuple[float, float] = (18.0, -60.0)) -> None:
        self._layout = layout
        self._vmin = vmin_db
        self._vmax = vmax_db
        self._view = view

    def _coords(self, kx: float, ky: float, range_pos: float) -> tuple[float, float, float]:
        """(kx,ky,range) -> (x,y,z) экрана по раскладке `AxisLayout` (VO решает, не мы)."""
        by_key = {"kx": kx, "ky": ky, "range": range_pos}
        return (by_key[self._layout.axis_x], by_key[self._layout.axis_y],
                by_key[self._layout.axis_z])

    def render(self, points: list[ScenePoint], markers: tuple[SceneMarker, ...] = (),
               title: str = "", figsize: tuple[float, float] = (9.0, 7.0),
               ax: plt.Axes | None = None) -> Figure:
        """Сцена: точки (цвет=дБ) + маркеры истины (чёрные ▲). `ax` — для врезок-панелей."""
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure
        if points:
            xyz = [self._coords(pt.kx, pt.ky, pt.range_pos) for pt in points]
            xs, ys, zs = zip(*xyz, strict=True)
            sc = ax.scatter(xs, ys, zs, c=[pt.db for pt in points], cmap="turbo",
                            vmin=self._vmin, vmax=self._vmax, s=40, alpha=0.85,
                            edgecolors="none")
            if ax.get_gid() != "no-colorbar":
                fig.colorbar(sc, ax=ax, shrink=0.7, label="дБ")
        for mk in markers:
            x, y, z = self._coords(mk.kx, mk.ky, mk.range_pos)
            ax.scatter([x], [y], [z], marker="^", color="k", s=70)
            ax.text(x, y, z, f" {mk.label}", fontsize=7)
        ax.set_xlabel(_LABELS[self._layout.axis_x], fontsize=8)
        ax.set_ylabel(_LABELS[self._layout.axis_y], fontsize=8)
        ax.set_zlabel(_LABELS[self._layout.axis_z], fontsize=8)
        if title:
            ax.set_title(title, fontsize=9)
        ax.view_init(*self._view)
        return fig
