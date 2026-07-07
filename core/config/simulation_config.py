"""Корневая конфигурация прогона + готовые сценарии."""
from __future__ import annotations

from dataclasses import dataclass, field

from .array_config import ArrayConfig, RangeConfig
from .scene_config import (
    BarrageSpec,
    DrfmCombSpec,
    HamEmitterSpec,
    SceneConfig,
    TargetSpec,
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


# --- Сценарии со смещением к краю по углу и ближе к концу по дальности ---------
# Оси куба: угол kx/ky in [-8..7] (0 = нормаль, край ~= +-6/7);
#           дальность 0..n_fft-1 = 0..63 (односторонняя, "конец" ~= 50+).
_BASE_ARRAY = ArrayConfig(16, 16)
_BASE_RANGE = RangeConfig(n_real=16, n_fft=64)


def target_edge_scenario() -> SimulationConfig:
    """1. Чистый ответ: одиночная точечная цель в углу кадра, у дальнего края."""
    scene = SceneConfig(
        emitters=(
            TargetSpec(kx=6, ky=6, range_bin=52, amplitude=1.0),
        ),
        thermal=ThermalNoiseSpec(power=0.02),
    )
    return SimulationConfig(array=_BASE_ARRAY, range=_BASE_RANGE, scene=scene, seed=7)


def barrage_edge_scenario() -> SimulationConfig:
    """2. Заградительная: шумовая заливка дальности с краевого направления.

    Заливает всю дальностную ось -> "конец" задаётся не бином, а краевым углом.
    """
    scene = SceneConfig(
        emitters=(
            BarrageSpec(kx=-7, ky=6, power=6.0),
        ),
        thermal=ThermalNoiseSpec(power=0.02),
    )
    return SimulationConfig(array=_BASE_ARRAY, range=_BASE_RANGE, scene=scene, seed=7)


def comb_edge_scenario() -> SimulationConfig:
    """3. Гребёнка DRFM: зубцы в дальнем полудиапазоне, под краевым углом.

    lead=30, spacing=6, count=5 -> бины 30,36,42,48,54 (ближе к концу оси 0..63).
    """
    scene = SceneConfig(
        emitters=(
            DrfmCombSpec(kx=6, ky=-6, lead_bin=30, spacing=6, count=5,
                         amplitude=1.0, decay=0.85),
        ),
        thermal=ThermalNoiseSpec(power=0.02),
    )
    return SimulationConfig(array=_BASE_ARRAY, range=_BASE_RANGE, scene=scene, seed=7)


def edge_scenarios() -> dict[str, SimulationConfig]:
    """Три краевых сценария по имени (для демо/датасета)."""
    return {
        "target": target_edge_scenario(),
        "barrage": barrage_edge_scenario(),
        "comb": comb_edge_scenario(),
    }
