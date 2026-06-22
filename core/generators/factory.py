"""Фабрика источников: спецификация (данные) -> объект-источник (поведение).

Abstract Factory с реестром: новый тип помехи регистрируется одной строкой,
без правки существующего кода (Open/Closed, Protected Variations, Indirection).
"""
from __future__ import annotations
from typing import Callable, Dict, Type

from ..config import (
    EmitterSpec, TargetSpec, DrfmCombSpec, BarrageSpec, HamEmitterSpec,
)
from .sources import SignalSource, PointTarget
from .jammers import DrfmComb, BarrageJammer, HamEmitter

Builder = Callable[[EmitterSpec], SignalSource]


class EmitterFactory:
    """Создаёт SignalSource по его спецификации (GRASP Creator)."""

    def __init__(self) -> None:
        self._builders: Dict[Type[EmitterSpec], Builder] = {}
        self._register_defaults()

    def register(self, spec_type: Type[EmitterSpec], builder: Builder) -> None:
        self._builders[spec_type] = builder

    def create(self, spec: EmitterSpec) -> SignalSource:
        try:
            return self._builders[type(spec)](spec)
        except KeyError as exc:
            raise ValueError(f"Нет билдера для {type(spec).__name__}") from exc

    def _register_defaults(self) -> None:
        self.register(TargetSpec, lambda s: PointTarget(
            s.kx, s.ky, s.range_bin, s.amplitude, s.phase))
        self.register(DrfmCombSpec, lambda s: DrfmComb(
            s.kx, s.ky, s.lead_bin, s.spacing, s.count, s.amplitude, s.decay))
        self.register(BarrageSpec, lambda s: BarrageJammer(s.kx, s.ky, s.power))
        self.register(HamEmitterSpec, lambda s: HamEmitter(
            s.kx, s.ky, s.amplitude, s.chirp_rate))
