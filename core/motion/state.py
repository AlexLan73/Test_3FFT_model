"""TargetState -- вектор состояния цели (Value Object, P1)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vec3(value: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    # np.array(..., copy=True) (дефолт) -- всегда новый буфер, в отличие от np.asarray
    # (который вернул бы view на входной np.ndarray -- VO протекал бы наружу, H1).
    arr = np.array(value, dtype=np.float64).reshape(3)
    arr.setflags(write=False)   # VO: состояние неизменяемо после конструирования
    return arr


@dataclass(frozen=True, eq=False)
class TargetState:
    """Позиция/скорость/ускорение цели (метры, м/с, м/с^2) + номер такта.

    `eq=False` (identity-семантика) -- как `SignalField`: numpy-массивы внутри
    делают дефолтный dataclass `__eq__` неоднозначным ("truth value of an array
    is ambiguous"), см. конвенцию `core/generators/waveforms`.
    """

    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tact: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", _vec3(self.pos))
        object.__setattr__(self, "vel", _vec3(self.vel))
        object.__setattr__(self, "acc", _vec3(self.acc))

    def evolved(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray) -> TargetState:
        """Новое состояние следующего такта (не мутирует текущее)."""
        return TargetState(pos=pos, vel=vel, acc=acc, tact=self.tact + 1)
