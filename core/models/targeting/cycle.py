"""Когнитивная петля такта (§8.3, Facade поверх готовых компонентов гейта/арбитра/целеуказания).

```
ЛЧМ широко -> 16x16xN (два FFT) -> токены + арбитр -> FM-m в выбранный квадрат -> пауза
```

Логика такта (§8.3): обнаружил пассивно (`VolumeTokenizer.tokenize` + `assemble_range`,
гл.4) -> дал точки (`Arbiter.arbitrate`, гл.5) -> зондировал активно (`Targeting.point`,
гл.8) -- куда светить пучком FM-m. Активный ОПРОС (сам зонд, приём отклика, гл.6-7)
выполняется вне этого класса -- он приходит с приёмника уже ПОСЛЕ того, как `beams`
сказали куда светить (см. `CodeArbiter`/`fm_correlate` в `arbiter.py` -- отдельный,
следующий такт петли). Здесь -- только "обнаружил -> дал точки -> указал", задел под
следующий такт (§8.3 "на следующем такте всё формируется заново").
"""
from __future__ import annotations

from dataclasses import dataclass

from ..result import SpectralCube
from ..tokenizer.arbiter import Arbiter, TargetDecision
from ..tokenizer.tokenizer import VolumeTokenizer, assemble_range
from ..tokenizer.tokens import RangeVerdict
from .beam import BeamCommand, Targeting


@dataclass(frozen=True)
class CycleResult:
    """Итог одного такта петли (VO, кросс-язычно -- только примитивы/кортежи VO).

    `verdicts`  -- проход 2 токенизатора (гл.4 §4.9, `assemble_range`).
    `decisions` -- метки арбитра (гл.5, `Arbiter.arbitrate`).
    `beams`     -- целеуказание (гл.8, `Targeting.point`) -- куда светить пучком FM-m.
    """

    verdicts: tuple[RangeVerdict, ...]
    decisions: tuple[TargetDecision, ...]
    beams: tuple[BeamCommand, ...]


class CognitiveCycle:
    """Facade §8.3: связывает готовые Strategy-компоненты в один такт петли (DI).

    Parameters
    ----------
    tokenizer     : `VolumeTokenizer` (гл.4) -- гейт, проход 1.
    edge_arbiter  : `Arbiter` (гл.5, `EdgeArbiter`/`CodeArbiter`/`CombinedArbiter`) --
                    метка цель/ложь.
    targeting     : `Targeting` (гл.8, `BeamTargeting`) -- целеуказание пучка.

    Компоненты создаются и настраиваются СНАРУЖИ (Composition Root) -- этот класс
    их не конструирует, только координирует такт (Facade, не Factory).
    """

    def __init__(self, tokenizer: VolumeTokenizer, edge_arbiter: Arbiter, targeting: Targeting) -> None:
        self._tokenizer = tokenizer
        self._arbiter = edge_arbiter
        self._targeting = targeting

    def step(self, cube: SpectralCube) -> CycleResult:
        """Один такт петли (§8.3): куб читается по ссылке, не мутируется ни здесь,
        ни в реюзаемых компонентах (`tokenize`/`assemble_range`/`arbitrate`/`point`)."""
        tokens = self._tokenizer.tokenize(cube)
        verdicts = assemble_range(tokens)
        decisions = self._arbiter.arbitrate(verdicts)
        beams = self._targeting.point(decisions)
        return CycleResult(verdicts=tuple(verdicts), decisions=tuple(decisions), beams=tuple(beams))
