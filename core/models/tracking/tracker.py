"""Трекер: связывает решения арбитра (`TargetDecision`, гл.5) МЕЖДУ тактами в
траектории (гл.4-бис §4.4, гл.5 §5.7) -- см. модульный докстринг `track.py`.

`Tracker(ABC)` -- Strategy: `update(decisions, tact) -> list[Track]` (активные
треки ПОСЛЕ такта). `NearestNeighborTracker` -- ближайший сосед в гейте
`(Δkx)^2 + (Δky)^2 + (w_r*Δlead_r)^2 < gate` (жадная ассоциация по возрастанию
расстояния, без венгерского алгоритма -- проще, детерминированно, для нашей
плотности целей достаточно).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..tokenizer.arbiter import TargetDecision
from .track import Track


class Tracker(ABC):
    """Strategy: список решений арбитра (один такт) -> активные треки после такта."""

    @abstractmethod
    def update(self, decisions: list[TargetDecision], tact: int) -> list[Track]:
        ...


@dataclass
class _TrackBook:
    """Мутабельная внутренняя бухгалтерия трека (НЕ отдаётся наружу -- наружу
    уходит иммутабельный снапшот `Track`, см. `_to_track`)."""

    track_id: int
    history: list[tuple[int, float, float, int]] = field(default_factory=list)
    missed: int = 0
    age: int = 0


def _linreg_slope(xs: list[int], ys: list[float]) -> float:
    """МНК-наклон `ys(xs)`. `< 2` точек -> 0.0 (скорость неизвестна, не выдумываем)."""
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x_mean = x.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0.0:
        return 0.0
    y_mean = y.mean()
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


class NearestNeighborTracker(Tracker):
    """Ближайший сосед в гейте -- см. модульный докстринг.

    Parameters
    ----------
    gate : порог квадрата гейт-расстояния `(Δkx)^2 + (Δky)^2 + (w_r*Δlead_r)^2`
        (сравнение БЕЗ извлечения корня -- дешевле, монотонно эквивалентно).
    w_r : вес дальностной компоненты гейта (дальность в бинах обычно на порядок
        крупнее углового дрейфа за такт -- без веса гейт пропускал бы либо
        слишком мало по углу, либо слишком много по дальности).
    max_missed : сколько тактов ПОДРЯД без ассоциации трек ещё жив; на
        `missed > max_missed` трек умирает (убирается из активных).
    moving_threshold : порог `sqrt(vel_r^2 + vel_angle^2)` (бин/такт) для «летит»
        (§4-бис.4). Дефолт калиброван на порядок дрейфа тестовых сценариев
        (статика ~0, движение ~10 бин/такт по дальности) -- не физическая
        константа патента, инженерный параметр Strategy.
    """

    def __init__(
        self,
        gate: float = 50.0,
        w_r: float = 0.3,
        max_missed: int = 2,
        moving_threshold: float = 1.5,
    ) -> None:
        if gate <= 0:
            raise ValueError(f"gate должен быть > 0, получено {gate}")
        if max_missed < 0:
            raise ValueError(f"max_missed должен быть >= 0, получено {max_missed}")
        self._gate = gate
        self._w_r = w_r
        self._max_missed = max_missed
        self._moving_threshold = moving_threshold
        self._books: dict[int, _TrackBook] = {}
        self._next_id = 0

    def update(self, decisions: list[TargetDecision], tact: int) -> list[Track]:
        targets = [d for d in decisions if d.decision == "target"]  # ложь/помеха не трекается

        # -- 1. кандидаты-пары (book, decision, d2) в гейте, по возрастанию d2.
        candidates: list[tuple[float, _TrackBook, TargetDecision]] = []
        for book in self._books.values():
            _, last_kx, last_ky, last_r = book.history[-1]
            for d in targets:
                d2 = (
                    (d.kx - last_kx) ** 2
                    + (d.ky - last_ky) ** 2
                    + (self._w_r * (d.lead_r - last_r)) ** 2
                )
                if d2 < self._gate:
                    candidates.append((d2, book, d))
        candidates.sort(key=lambda c: c[0])

        # -- 2. жадная ассоциация: ближайшая пара первой, каждый book/decision -- один раз.
        matched_books: set[int] = set()
        matched_decisions: set[int] = set()
        assoc: dict[int, TargetDecision] = {}  # track_id -> decision
        for _, book, d in candidates:
            if book.track_id in matched_books or id(d) in matched_decisions:
                continue
            matched_books.add(book.track_id)
            matched_decisions.add(id(d))
            assoc[book.track_id] = d

        # -- 3. обновить сматченные / состарить несматченные (пропуск/смерть).
        dead_ids: list[int] = []
        for book in self._books.values():
            book.age += 1
            d = assoc.get(book.track_id)
            if d is not None:
                book.history.append((tact, d.kx, d.ky, d.lead_r))
                book.missed = 0
            else:
                book.missed += 1
                if book.missed > self._max_missed:
                    dead_ids.append(book.track_id)
        for tid in dead_ids:
            del self._books[tid]

        # -- 4. рождение: несматченные target-декларации -> новые треки.
        for d in targets:
            if id(d) in matched_decisions:
                continue
            book = _TrackBook(track_id=self._next_id, history=[(tact, d.kx, d.ky, d.lead_r)],
                               missed=0, age=1)
            self._books[book.track_id] = book
            self._next_id += 1

        return [self._to_track(book) for book in self._books.values()]

    def _to_track(self, book: _TrackBook) -> Track:
        tacts = [h[0] for h in book.history]
        kxs = [h[1] for h in book.history]
        kys = [h[2] for h in book.history]
        rs = [float(h[3]) for h in book.history]

        vel_r = _linreg_slope(tacts, rs)
        vel_kx = _linreg_slope(tacts, kxs)
        vel_ky = _linreg_slope(tacts, kys)
        vel_angle = float(np.hypot(vel_kx, vel_ky))
        speed = float(np.hypot(vel_r, vel_angle))

        _, last_kx, last_ky, last_r = book.history[-1]
        return Track(
            track_id=book.track_id,
            kx=last_kx,
            ky=last_ky,
            lead_r=last_r,
            history=tuple(book.history),
            vel_r=vel_r,
            vel_angle=vel_angle,
            age=book.age,
            missed=book.missed,
            is_moving=speed > self._moving_threshold,
        )
