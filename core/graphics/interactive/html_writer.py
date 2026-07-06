"""Запись plotly-фигур на диск -- Pure Fabrication (IO отделён от рендера)."""
from __future__ import annotations

import os

import plotly.graph_objects as go


class HtmlWriter:
    def __init__(self, out_dir: str) -> None:
        self._dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write(self, fig: go.Figure, name: str) -> str:
        path = os.path.join(self._dir, name)
        fig.write_html(path, include_plotlyjs=True, full_html=True)
        return path
