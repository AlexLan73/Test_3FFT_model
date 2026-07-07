"""Фабрика источников: спецификация (данные) -> объект-источник (поведение).

Abstract Factory с реестром: новый тип помехи регистрируется одной строкой,
без правки существующего кода (Open/Closed, Protected Variations, Indirection).
"""
from __future__ import annotations

from collections.abc import Callable

from ..config import (
    BarrageSpec,
    DrfmCombSpec,
    EmitterSpec,
    HamEmitterSpec,
    TargetSpec,
)
from .jammers import BarrageJammer, DrfmComb, HamEmitter
from .sources import PointTarget, SignalSource

Builder = Callable[[EmitterSpec], SignalSource]


class EmitterFactory:
    """Создаёт SignalSource по его спецификации (GRASP Creator)."""

    def __init__(self) -> None:
        self._builders: dict[type[EmitterSpec], Builder] = {}
        self._register_defaults()

    def register(self, spec_type: type[EmitterSpec], builder: Builder) -> None:
        self._builders[spec_type] = builder

    def create(self, spec: EmitterSpec) -> SignalSource:
        try:
            return self._builders[type(spec)](spec)
        except KeyError as exc:
            raise ValueError(f"Нет билдера для {type(spec).__name__}") from exc

    def _register_defaults(self) -> None:
        def build_target(s: EmitterSpec) -> SignalSource:
            assert isinstance(s, TargetSpec)
            return PointTarget(s.kx, s.ky, s.range_bin, s.amplitude, s.phase)

        def build_drfm(s: EmitterSpec) -> SignalSource:
            assert isinstance(s, DrfmCombSpec)
            return DrfmComb(s.kx, s.ky, s.lead_bin, s.spacing, s.count, s.amplitude, s.decay)

        def build_barrage(s: EmitterSpec) -> SignalSource:
            assert isinstance(s, BarrageSpec)
            return BarrageJammer(s.kx, s.ky, s.power)

        def build_ham(s: EmitterSpec) -> SignalSource:
            assert isinstance(s, HamEmitterSpec)
            return HamEmitter(s.kx, s.ky, s.amplitude, s.chirp_rate)

        self.register(TargetSpec, build_target)
        self.register(DrfmCombSpec, build_drfm)
        self.register(BarrageSpec, build_barrage)
        self.register(HamEmitterSpec, build_ham)
