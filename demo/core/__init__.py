"""demo.core — общий стенд для примеров (Pure Fabrication).

Наполняется по мере постановки задач: runner (Template Method) + report +
writer готовы (ex1). scenes/placement/inspect — с ex2.
"""
from __future__ import annotations

from .report import DemoReport
from .runner import DemoContext, DemoRunner
from .writer import DemoWriter

__all__ = ["DemoRunner", "DemoContext", "DemoWriter", "DemoReport"]
