"""Абстракция классификатора куба (Strategy).

Отдельная ответственность от RadarModel: тот делает спектральный куб,
этот выносит по нему решение. Любая реализация взаимозаменяема (LSP).
"""
from __future__ import annotations
from abc import ABC, abstractmethod

from ..result import SpectralCube
from .labels import Classification


class CubeClassifier(ABC):
    @abstractmethod
    def classify(self, cube: SpectralCube) -> Classification:
        ...
