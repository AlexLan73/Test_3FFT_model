"""TactSequence -- минимальный итератор тактов движения одной цели (Iterator, P1/A3).

P1 -- фундамент: одна цель, без генерации куба (это P2) и без мульти-цели (P4).
Реюз `Scene`/`Synthesizer` (`core/generators/scene.py`) относится к P2 (когда
трек начнёт заполнять вход куба) -- здесь они не нужны и не импортируются, чтобы
не тащить фиктивную зависимость (A1: `Scene` не плодим второй раз, но и не
привязываем к нему то, что его не касается).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..data_context import DataContext
from ..motion import Kinematics, KinematicsSample, MotionModel, TargetState

TRACK_CHANNEL = "tracks"


@dataclass(frozen=True, eq=False)
class Tact:
    """Одна запись трека: состояние цели ДО шага + её проекция в бины куба."""

    state: TargetState
    sample: KinematicsSample


class TactSequence(Iterator[Tact]):
    """Итератор: `n_tacts` тактов движения одной цели через `MotionModel`.

    На каждом такте публикует `Tact` в `DataContext.publish(TRACK_CHANNEL, ...)`,
    если контекст передан (SPEC §4: любой обмен -- через `DataContext`/шину, без
    прямого I/O в обход). Без `DataContext` работает как обычный итератор трека.
    """

    def __init__(self, initial: TargetState, model: MotionModel, kinematics: Kinematics,
                 n_tacts: int, dt: float = 1.0, rng: np.random.Generator | None = None,
                 data_context: DataContext | None = None) -> None:
        if n_tacts < 0:
            raise ValueError("n_tacts не может быть отрицательным")
        self._state = initial
        self._model = model
        self._kinematics = kinematics
        self._n_tacts = n_tacts
        self._dt = dt
        self._rng = rng if rng is not None else np.random.default_rng()
        self._data = data_context
        self._i = 0

    def __iter__(self) -> Iterator[Tact]:
        return self

    def __next__(self) -> Tact:
        if self._i >= self._n_tacts:
            raise StopIteration
        sample = self._kinematics.project(self._state, self._dt)
        tact = Tact(state=self._state, sample=sample)
        if self._data is not None:
            self._data.publish(TRACK_CHANNEL, tact)
        self._state = self._model.propagate(self._state, self._dt, self._rng)
        self._i += 1
        return tact
