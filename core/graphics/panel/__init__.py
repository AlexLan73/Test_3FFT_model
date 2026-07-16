"""core.graphics.panel -- живая панель управления сценой (Observer над `Transport`, P6).

Две части (N5, GUI-free тестируемость): `panel_model.py` -- чистая дата-модель
(`Field`/`Cell`/`Element`/`SignalBlock`, закладка +-N, БЕЗ импорта dearpygui, целиком
покрыта `TestRunner`-тестами) и `panel_app.py` -- тонкая обвязка Dear PyGui поверх
неё (импорт `dearpygui` под `try/except`, недоступность библиотеки/дисплея не роняет
импорт пакета).
"""
from __future__ import annotations

from .panel_model import Cell, Field, PanelModel, SignalBlock, lerp_field

__all__ = ["Cell", "Field", "PanelModel", "SignalBlock", "lerp_field"]
