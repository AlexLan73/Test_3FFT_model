"""TactSequence -- минимальный итератор тактов движения одной цели (Iterator, P1/A3).

P1 -- фундамент: одна цель, без генерации куба (это P2) и без мульти-цели (P4).
Реюз `Scene`/`Synthesizer` (`core/generators/scene.py`) относится к P2 (когда
трек начнёт заполнять вход куба) -- здесь они не нужны и не импортируются, чтобы
не тащить фиктивную зависимость (A1: `Scene` не плодим второй раз, но и не
привязываем к нему то, что его не касается).

P4 (M2, см. `MemoryBank/tasks/TASK_body_motion_p4.md`): `MultiTactSequence` -- координатор
НЕСКОЛЬКИХ независимых целей, добавлен РЯДОМ, `TactSequence` (одноцелевой, P1) не трогаем
(не ломаем `tests/test_body_motion.py`) -- реюзует тот же `Tact`/`Kinematics.project`/
`MotionModel.propagate`, просто по списку `(TargetState, MotionModel)` со своим ГСЧ на цель.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
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


@dataclass(frozen=True)
class TargetHandle:
    """P4: одна цель мульти-сцены -- начальное состояние + закон движения + свой ГСЧ.

    `seed` (M4, сверка Кодо): у каждой цели -- СВОЙ `np.random.Generator` (см.
    `MultiTactSequence.__init__`), стохастические модели (`MarkovDrift`) разных целей
    не коррелируют и не делят общее состояние ГСЧ между собой. `seed=None` -> генератор
    сам себе берёт случайную энтропию (как `TactSequence` без явного `rng`).
    """

    initial: TargetState
    model: MotionModel
    seed: int | None = None


@dataclass(frozen=True, eq=False)
class MultiTact:
    """Одна запись мульти-трека такта: `Tact` каждой цели, порядок -- как в `targets`."""

    tacts: tuple[Tact, ...]


class MultiTactSequence(Iterator[MultiTact]):
    """Итератор: `n_tacts` тактов движения НЕСКОЛЬКИХ независимых целей (P4, M2).

    Каждая цель продвигается своим `MotionModel` через `Kinematics.project`/
    `MotionModel.propagate` -- ТОТ ЖЕ код, что `TactSequence`, просто по списку
    `TargetHandle`. На каждом такте публикует `MultiTact` (кортеж `Tact` по числу целей)
    в `DataContext.publish(TRACK_CHANNEL, ...)`, если контекст передан (тот же канал
    "tracks", что у `TactSequence` -- наблюдателю достаточно проверить тип `data`,
    чтобы отличить одно-/мульти-трек публикацию).
    """

    def __init__(self, targets: Sequence[TargetHandle], kinematics: Kinematics,
                 n_tacts: int, dt: float = 1.0,
                 data_context: DataContext | None = None) -> None:
        if n_tacts < 0:
            raise ValueError("n_tacts не может быть отрицательным")
        if len(targets) == 0:
            raise ValueError("targets не может быть пустым -- нечего двигать")
        self._states: list[TargetState] = [t.initial for t in targets]
        self._models: list[MotionModel] = [t.model for t in targets]
        # M4: свой независимый ГСЧ на каждую цель (не общий rng на всех).
        self._rngs: list[np.random.Generator] = [np.random.default_rng(t.seed) for t in targets]
        self._kinematics = kinematics
        self._n_tacts = n_tacts
        self._dt = dt
        self._data = data_context
        self._i = 0

    def __len__(self) -> int:
        """Число целей в сцене (не число тактов -- см. `n_tacts` в конструкторе)."""
        return len(self._states)

    def __iter__(self) -> Iterator[MultiTact]:
        return self

    def __next__(self) -> MultiTact:
        if self._i >= self._n_tacts:
            raise StopIteration
        tacts: list[Tact] = []
        for idx, state in enumerate(self._states):
            sample = self._kinematics.project(state, self._dt)
            tacts.append(Tact(state=state, sample=sample))
            self._states[idx] = self._models[idx].propagate(state, self._dt, self._rngs[idx])
        multi = MultiTact(tacts=tuple(tacts))
        if self._data is not None:
            self._data.publish(TRACK_CHANNEL, multi)
        self._i += 1
        return multi
