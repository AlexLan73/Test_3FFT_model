"""Корневая конфигурация прогона + готовые сценарии."""
from __future__ import annotations
from dataclasses import dataclass, field

from .array_config import ArrayConfig, RangeConfig
from .scene_config import (
    SceneConfig, TargetSpec, DrfmCombSpec, BarrageSpec, HamEmitterSpec,
    ThermalNoiseSpec,
)


@dataclass(frozen=True)
class SimulationConfig:
    """Всё, что нужно для воспроизводимого прогона модели."""
    array: ArrayConfig = field(default_factory=ArrayConfig)
    range: RangeConfig = field(default_factory=RangeConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    seed: int = 7


def default_scenario() -> SimulationConfig:
    """Эталонная сцена: цель + гребёнка DRFM на нормали, радиолюбитель сбоку."""
    scene = SceneConfig(
        emitters=(
            TargetSpec(kx=0, ky=0, range_bin=8, amplitude=1.0),
            DrfmCombSpec(kx=0, ky=0, lead_bin=8, spacing=6, count=5, amplitude=1.0),
            HamEmitterSpec(kx=5, ky=-3, amplitude=5.0),
        ),
        thermal=ThermalNoiseSpec(power=0.02),
    )
    return SimulationConfig(
        array=ArrayConfig(16, 16),
        range=RangeConfig(n_real=16, n_fft=64),
        scene=scene,
        seed=7,
    )
