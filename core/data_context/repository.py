"""Хранилище кубов данных (load/save). Абстракция + numpy-реализация."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class CubeRepository(ABC):
    @abstractmethod
    def save(self, name: str, cube: np.ndarray) -> str: ...

    @abstractmethod
    def load(self, name: str) -> np.ndarray: ...


class NpyCubeRepository(CubeRepository):
    """Сохранение/загрузка комплексных кубов в формате .npy."""

    def __init__(self, root: str):
        self._root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self._root, name if name.endswith(".npy") else name + ".npy")

    def save(self, name, cube):
        path = self._path(name)
        np.save(path, cube)
        return path

    def load(self, name):
        return np.load(self._path(name))
