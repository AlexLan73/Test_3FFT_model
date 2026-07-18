"""Целеуказание пучка FM-m (гл.8, `Doc/Patent/glava8_celeukazanie.md`) -- публичный API пакета."""
from .beam import BeamCommand, BeamTargeting, Targeting
from .cycle import CognitiveCycle, CycleResult

__all__ = ["BeamCommand", "Targeting", "BeamTargeting", "CognitiveCycle", "CycleResult"]
