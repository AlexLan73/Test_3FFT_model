"""Целеуказание пучка FM-m (гл.8, `Doc/Patent/glava8_celeukazanie.md`).

Гейт (гл.4, токенизатор) свернул объём в матрицу токенов, арбитр (гл.5,
`arbiter.py`) пометил переднего края «цель/ложь». Настоящий модуль -- шаг (г)
патента: из решений арбитра (грубые `(r, kx, ky)` истинных целей) формируется
**целеуказание** -- куда и каким пучком светить агильным FM-m (§8.2).

Вокруг грубой оценки остаётся КОНУС неопределённости (дискретность угловой
сетки, страддл, шум, §8.2) -- в него направляется пучок из нескольких лучей
(гл.6 §6, 5-7 штук), покрывающих конус, а не один луч точно "в яблочко".
Тонкую дальность и подтверждение даёт уже сам опрос (гл.6-7) -- ЗДЕСЬ только
"куда светить", не результат опроса.

Работает по ЕДИНИЦАМ токенов (`TargetDecision`), не по кубу (§8.4) -- решение
принимается по нескольким числам на цель, а не по карте/кубу.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..tokenizer.arbiter import TargetDecision


@dataclass(frozen=True)
class BeamCommand:
    """Команда "куда светить" (VO, кросс-язычно под C++/msgpack -- только примитивы).

    `target_r`   -- дальность цели-кандидата (передний край, `TargetDecision.lead_r`).
    `center_kx`, `center_ky` -- центр конуса (грубая угловая оценка арбитра).
    `beam_angles` -- углы лучей пучка `(kx, ky)`, покрывающих конус вокруг центра
                     (§8.2, гл.6 §6: 5-7 лучей); включает сам центр. Порядок --
                     по возрастанию расстояния до центра, детерминирован (стабильная
                     сортировка по `(dist, dkx, dky)`).
    """

    target_r: int
    center_kx: float
    center_ky: float
    beam_angles: tuple[tuple[float, float], ...]


class Targeting(ABC):
    """Strategy §8: решения арбитра (`TargetDecision`, только `decision=="target"`) ->
    команды пучка (`BeamCommand`). По кубу НЕ работает (§8.4) -- только по токенам/декларациям.
    """

    @abstractmethod
    def point(self, decisions: list[TargetDecision]) -> list[BeamCommand]:
        ...


class BeamTargeting(Targeting):
    """Многолучевой пучок вокруг конуса неопределённости (§8.2).

    Parameters
    ----------
    cone_half : полуширина конуса в бинах угловой сетки (шаг `step`) по каждой
                оси -- лучи покрывают `[-cone_half, +cone_half]` вокруг центра.
    max_beams : верхняя граница числа лучей на цель (гл.6 §6: 5-7) -- после
                сортировки по расстоянию до центра берём первые `max_beams`.
    step      : угловой шаг луча (в тех же единицах, что `kx`/`ky` декларации).

    Только `decision == "target"` порождает команду -- по `jammer`/`false`
    светить незачем (§8: "излучают в выбранные площади" -- площади целей).
    """

    def __init__(self, cone_half: int = 1, max_beams: int = 7, step: float = 1.0) -> None:
        if cone_half < 0:
            raise ValueError(f"cone_half должен быть >= 0, получено {cone_half}")
        if max_beams < 1:
            raise ValueError(f"max_beams должен быть >= 1, получено {max_beams}")
        if step <= 0.0:
            raise ValueError(f"step должен быть > 0, получено {step}")
        self._cone_half = cone_half
        self._max_beams = max_beams
        self._step = step

    def point(self, decisions: list[TargetDecision]) -> list[BeamCommand]:
        commands: list[BeamCommand] = []
        for d in decisions:
            if d.decision != "target":
                continue  # jammer/false -- не светим (§8: только по целям)
            angles = self._cone_beam_angles(d.kx, d.ky)
            commands.append(BeamCommand(
                target_r=d.lead_r, center_kx=d.kx, center_ky=d.ky, beam_angles=angles,
            ))
        return commands

    def _cone_beam_angles(self, center_kx: float, center_ky: float) -> tuple[tuple[float, float], ...]:
        n = self._cone_half
        offsets = [
            (dkx * self._step, dky * self._step)
            for dkx in range(-n, n + 1)
            for dky in range(-n, n + 1)
        ]
        # По возрастанию расстояния до центра, детерминированная стабильная сортировка
        # (центр -- первый, дальше -- ближайшие соседи конуса).
        offsets.sort(key=lambda o: (o[0] ** 2 + o[1] ** 2, o[0], o[1]))
        offsets = offsets[: self._max_beams]
        return tuple((center_kx + dkx, center_ky + dky) for dkx, dky in offsets)
