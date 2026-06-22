"""Спецификации источников сцены и сама сцена (Value Objects).

Спецификации -- это чистые данные (что разместить), отделённые от их синтеза.
Конкретные классы-источники строит EmitterFactory (см. generators/factory.py).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class EmitterSpec:
    """Базовая спецификация излучателя/отражателя."""
    kx: float = 0.0          # угловой бин (азимут), 0 = нормаль
    ky: float = 0.0          # угловой бин (угол места)
    amplitude: float = 1.0


@dataclass(frozen=True)
class TargetSpec(EmitterSpec):
    """Истинная точечная цель на дальности range_bin (несёт текущий код)."""
    range_bin: float = 8.0
    phase: float = 0.0


@dataclass(frozen=True)
class DrfmCombSpec(EmitterSpec):
    """Гребёнка ложных целей DRFM: передний фронт + копии ТОЛЬКО позади."""
    lead_bin: float = 8.0
    spacing: float = 6.0
    count: int = 5
    decay: float = 0.85       # спад амплитуды от зубца к зубцу


@dataclass(frozen=True)
class BarrageSpec(EmitterSpec):
    """Заградительная (шумовая) помеха с одного направления: заливка дальности."""
    power: float = 6.0


@dataclass(frozen=True)
class HamEmitterSpec(EmitterSpec):
    """Стороннее излучение (радиолюбитель): после дерампа -- размаз по дальности."""
    chirp_rate: Optional[float] = None   # None -> авто (полный размах по полосе)


@dataclass(frozen=True)
class ThermalNoiseSpec:
    """Тепловой шум приёмника: независим по элементам, без направления."""
    power: float = 0.02


@dataclass(frozen=True)
class SceneConfig:
    """Набор спецификаций, описывающих обстановку (цели + помехи + шум)."""
    emitters: Sequence[EmitterSpec] = field(default_factory=tuple)
    thermal: ThermalNoiseSpec = field(default_factory=ThermalNoiseSpec)
