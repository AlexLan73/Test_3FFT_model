"""Фабрика размеченных кубов: генератор сцен -> (куб, метка).

Использует тот же синтез, что и боевой тракт, поэтому метки идеальны и
данных можно сделать сколько угодно. Single Responsibility: только данные.
"""
from __future__ import annotations

import numpy as np

from ...config import (
    ArrayConfig,
    BarrageSpec,
    DrfmCombSpec,
    EmitterSpec,
    HamEmitterSpec,
    RangeConfig,
    SceneConfig,
    TargetSpec,
    ThermalNoiseSpec,
)
from ...generators import SceneBuilder, Synthesizer
from ..base import RadarModel
from .labels import CLASS_NAMES


class CubeDatasetGenerator:
    """Генерирует одно-классовые кубы со случайными параметрами для обучения."""

    def __init__(self, array: ArrayConfig, rng: RangeConfig, model: RadarModel,
                 seed: int = 0):
        self._array = array
        self._range = rng
        self._model = model
        self._rs = np.random.default_rng(seed)
        self._builder = SceneBuilder()

    def _scene_for(self, name: str) -> SceneConfig:
        rs = self._rs
        kx, ky = float(rs.uniform(-6, 6)), float(rs.uniform(-6, 6))
        thermal = ThermalNoiseSpec(power=0.02)
        emitters: tuple[EmitterSpec, ...] = ()
        if name == "empty":
            pass
        elif name == "target":
            emitters = (TargetSpec(kx=kx, ky=ky, range_bin=float(rs.uniform(4, 40)),
                                   amplitude=float(rs.uniform(0.6, 1.2))),)
        elif name == "barrage":
            emitters = (BarrageSpec(kx=kx, ky=ky, power=float(rs.uniform(3, 8))),)
        elif name == "comb":
            emitters = (DrfmCombSpec(kx=kx, ky=ky, lead_bin=float(rs.uniform(4, 20)),
                                     spacing=float(rs.uniform(4, 8)),
                                     count=int(rs.integers(3, 6)),
                                     amplitude=float(rs.uniform(0.7, 1.1))),)
        elif name == "ham":
            emitters = (HamEmitterSpec(kx=kx, ky=ky,
                                       amplitude=float(rs.uniform(3, 6))),)
        else:
            raise ValueError(f"неизвестный класс {name}")
        return SceneConfig(emitters=emitters, thermal=thermal)

    def sample(self, name: str) -> tuple[np.ndarray, int]:
        scene = self._builder.build(self._scene_for(name))
        seed = int(self._rs.integers(0, 2 ** 31))      # своя реализация шума
        raw = Synthesizer(self._array, self._range, seed).build(scene)
        cube = self._model.process(raw).magnitude.astype("float32")
        return cube, CLASS_NAMES.index(name)

    def batch(self, n: int, balanced: bool = True) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(n):
            name = CLASS_NAMES[i % len(CLASS_NAMES)] if balanced \
                else str(self._rs.choice(CLASS_NAMES))
            cube, label = self.sample(name)
            xs.append(cube)
            ys.append(label)
        x_batch = np.stack(xs)[:, None]                 # (n, 1, nx, ny, n_fft)
        return x_batch.astype("float32"), np.asarray(ys, dtype="int64")
