"""Интерактивная ветка (plotly) -- параллельна matplotlib-ветке `core.graphics`.

Импортируется отдельно (`core.graphics.interactive`), чтобы `import core.graphics`
не тянул plotly (мягкая зависимость, см. F3 в спеке graphics_refactor_2026-07-06).
"""
from .cube_interactive import InteractiveCubeVisualizer
from .html_writer import HtmlWriter
from .interactive_visualizer import InteractiveVisualizer

__all__ = ["InteractiveVisualizer", "InteractiveCubeVisualizer", "HtmlWriter"]
