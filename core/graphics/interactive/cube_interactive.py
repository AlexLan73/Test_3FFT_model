"""Интерактивный 3D-скаттер куба (plotly): слайдер порога = кадры видимости.

Переиспользует `CubeSampler`/`AxisLayout` из `core.graphics` (общий код с
matplotlib-веткой), но собственный рендер -- `plotly.graph_objects.Figure`.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import plotly.graph_objects as go

from ...models import SpectralCube
from ..layout import AxisLayout
from ..sampling import CubeSampler
from .interactive_visualizer import InteractiveVisualizer

# B008: вызов classmethod в дефолте аргумента → module-level синглтон (frozen dataclass, безопасно).
_LAYOUT_DEPTH: AxisLayout = AxisLayout.range_in_depth()


class InteractiveCubeVisualizer(InteractiveVisualizer):
    """layout -- раскладка осей; thresholds -- набор порогов (дБ) для слайдера;
    default_db -- активный кадр при открытии (если не входит в thresholds --
    берётся ближайший, F6: иначе thresholds.index(default_db) кидает ValueError)."""

    def __init__(self, layout: AxisLayout = _LAYOUT_DEPTH,
                 thresholds: Iterable[float] = range(-40, -5, 2),
                 default_db: float = -22) -> None:
        self._layout = layout
        self._thresholds = tuple(thresholds)
        if default_db not in self._thresholds:
            default_db = min(self._thresholds, key=lambda t: abs(t - default_db))
        self._default_db = default_db

    @property
    def default_db(self) -> float:
        """Фактически выбранный активный порог (после гарда F6)."""
        return self._default_db

    def render(self, cube: SpectralCube) -> go.Figure:
        layout = self._layout
        _, xlabel, xlim = layout.resolve(cube, layout.axis_x)
        _, ylabel, ylim = layout.resolve(cube, layout.axis_y)
        _, zlabel, zlim = layout.resolve(cube, layout.axis_z)

        traces = []
        for thr in self._thresholds:
            pts = CubeSampler(threshold_db=thr).points(cube, layout)
            traces.append(go.Scatter3d(
                x=pts.x, y=pts.y, z=pts.z,
                mode="markers",
                marker=dict(
                    size=np.clip((pts.values_db - thr) * 0.22 + 1.5, 1.5, 9),
                    color=pts.values_db, colorscale="Turbo",
                    cmin=self._default_db, cmax=0,
                    opacity=0.75,
                    colorbar=dict(title="дБ отн. макс", len=0.6),
                ),
                hovertemplate=(f"{xlabel}=%{{x}}<br>{ylabel}=%{{y}}<br>{zlabel}=%{{z}}"
                              "<br>%{marker.color:.1f} дБ<extra></extra>"),
                visible=(thr == self._default_db),
                name=f"{thr} дБ",
            ))

        fig = go.Figure(data=traces)

        steps = []
        for i, thr in enumerate(self._thresholds):
            steps.append(dict(method="update", args=[{"visible": [j == i for j in range(len(traces))]}],
                              label=f"{thr}"))

        fig.update_layout(
            title="3D-БПФ куб (интерактив): порог отсева -- слайдер снизу",
            scene=dict(
                xaxis=dict(title=xlabel, range=list(xlim)),
                yaxis=dict(title=ylabel, range=list(ylim)),
                zaxis=dict(title=zlabel, range=list(zlim)),
            ),
            sliders=[dict(
                active=self._thresholds.index(self._default_db),
                currentvalue=dict(prefix="Порог отсева: ", suffix=" дБ отн. макс"),
                pad=dict(t=40), steps=steps,
            )],
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig
