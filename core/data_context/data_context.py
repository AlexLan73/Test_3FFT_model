"""Фасад доступа к данным: единая точка входа для load/save."""
from __future__ import annotations
import numpy as np

from .repository import CubeRepository, NpyCubeRepository


class DataContext:
    """Facade над репозиториями. Скрывает детали хранения от контроллера."""

    def __init__(self, repository: CubeRepository | None = None,
                 root: str = "./data"):
        self._repo = repository or NpyCubeRepository(root)

    def save_cube(self, name: str, cube: np.ndarray) -> str:
        return self._repo.save(name, cube)

    def load_cube(self, name: str) -> np.ndarray:
        return self._repo.load(name)
