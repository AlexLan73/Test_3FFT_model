"""Сцена (Composite) + строитель сцены + синтезатор куба данных."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..config import ArrayConfig, RangeConfig, SceneConfig
from .factory import EmitterFactory
from .grid import ArrayGrid
from .sources import SignalSource, ThermalNoise


class Scene(SignalSource):
    """Композит источников: вклад сцены = сумма вкладов составляющих."""

    def __init__(self, sources: Iterable[SignalSource] | None = None):
        self._sources: list[SignalSource] = list(sources or [])

    def add(self, source: SignalSource) -> Scene:
        self._sources.append(source)
        return self

    def contribute(self, grid, rng, rs):
        acc = self._empty(grid, rng)
        for src in self._sources:
            acc = acc + src.contribute(grid, rng, rs)
        return acc


class SceneBuilder:
    """Строит Scene из SceneConfig через фабрику (Builder)."""

    def __init__(self, factory: EmitterFactory | None = None):
        self._factory = factory or EmitterFactory()

    def build(self, cfg: SceneConfig) -> Scene:
        scene = Scene()
        for spec in cfg.emitters:
            scene.add(self._factory.create(spec))
        scene.add(ThermalNoise(cfg.thermal.power))   # шум добавляем последним
        return scene


class Synthesizer:
    """Синтезирует сырой куб (nx, ny, n_real) из сцены. Single Responsibility."""

    def __init__(self, array: ArrayConfig, rng: RangeConfig, seed: int = 7):
        self._grid = ArrayGrid.from_config(array)
        self._rng = rng
        self._rs = np.random.default_rng(seed)

    def build(self, scene: Scene) -> np.ndarray:
        return scene.contribute(self._grid, self._rng, self._rs)
