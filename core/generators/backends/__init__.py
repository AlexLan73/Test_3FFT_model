"""Подпакет бэкендов генерации (Strategy, §4.3 спеки). Реэкспорт публичного API."""
from __future__ import annotations

from .base import GenBackend
from .numpy_backend import NumpyBackend

__all__ = ["GenBackend", "NumpyBackend"]
