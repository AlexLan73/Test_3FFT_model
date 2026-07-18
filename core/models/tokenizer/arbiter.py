"""Физический арбитр гл.5 (`Doc/Patent/glava5_peredniy_krai.md`) -- передний край, τ≥0.

Гейт (гл.4, токенизатор) отбирает кандидатов и грубо типизирует причинностную
группу (`target`/`comb`/`barrage`, `assemble_range`), но метку «цель/ложь» НЕ
ставит -- ретранслятор даёт «собранный источник», по одной угловой карте
неотличимый от настоящей цели. Метку ставит арбитр этого модуля, по причинности:

    τ ≥ 0  →  R_ложн ≥ R_цель   (§5.1)

ретранслятор обязан сначала принять зонд, потом переизлучить -- его копии
всегда на ТОЙ ЖЕ дальности или дальше истинной цели, никогда ближе. Решение --
**по дальности, не по яркости** (ложная копия может быть ярче).

`EdgeArbiter` -- Вариант 1 (§5.2, геометрия): ведущий (ближний, `lead_r`) член
причинностной группы -- истинная цель. `CodeArbiter` -- Вариант 2 (§5.3,
свежесть FM-m кода) -- задел, реализация после FM-m опроса (гл.6).

Область действия (§5.5): правило `barrage_selfscreen` строго для
САМОЗАЩИТНОЙ помехи (носитель = цель, тот же угол) -- джиттер-осколки
`target`-вердиктов под соседними углами (пик заграда "уезжает" в соседний бин,
`MemoryBank/specs/tokenizer_barrage_pass2_2026-07-17.md`) поглощаются заградом
угловой толерантностью, а не порождают ложные target-decision'ы.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .tokens import BARRAGE, COMB, RangeVerdict


@dataclass(frozen=True)
class TargetDecision:
    """Финальная метка арбитра (VO, кросс-язычно под C++/msgpack -- только примитивы).

    `lead_r`   -- передний край причинностной группы (дальность истинной цели-кандидата).
    `decision` -- "target" | "false" | "jammer".
    `reason`   -- "single" | "leading_edge" | "comb_tail" | "barrage_selfscreen".
    `n_false`  -- число ложных копий за передним краем (оценка, для comb; 0 если неизвестно).
    """

    kx: float
    ky: float
    lead_r: int
    decision: str
    reason: str
    n_false: int = 0


class Arbiter(ABC):
    """Strategy: `list[RangeVerdict]` (проход 2 токенизатора) -> финальные метки цель/ложь."""

    @abstractmethod
    def arbitrate(self, verdicts: list[RangeVerdict]) -> list[TargetDecision]:
        ...


class EdgeArbiter(Arbiter):
    """Вариант 1 (§5.2): геометрия τ≥0 -- передний край группы по дальности.

    Parameters
    ----------
    angle_merge_tol : угловая толерантность (в бинах, §5.5) для поглощения
                      джиттер-осколков `target` соседним `barrage` -- реальный
                      заград джиттерит по углу, часть его энергии иногда
                      триажится как отдельный `source` под соседним (kx,ky) и
                      собирается проходом 2 в одиночный "target"-вердикт; это
                      осколок ТОЙ ЖЕ самозащитной помехи, не отдельная цель.
    """

    def __init__(self, angle_merge_tol: int = 1) -> None:
        if angle_merge_tol < 0:
            raise ValueError(f"angle_merge_tol должен быть >= 0, получено {angle_merge_tol}")
        self._tol = angle_merge_tol

    def arbitrate(self, verdicts: list[RangeVerdict]) -> list[TargetDecision]:
        barrages = [v for v in verdicts if v.kind == BARRAGE]

        decisions: list[TargetDecision] = []
        for v in verdicts:
            if v.kind == BARRAGE:
                # §5.5: заград доминирует -- кандидат-цель под самоприкрытием
                # сохраняется в lead_r, но метка -- jammer (заградка не отделяет
                # своего носителя от цели по дальности на этом уровне).
                decisions.append(TargetDecision(
                    kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                    decision="jammer", reason="barrage_selfscreen", n_false=0,
                ))
                continue

            if v.kind == COMB:
                # §5.2: ведущий (ближний) член гребёнки -- истинная цель; хвостовые
                # регулярные копии -- ложные, но НЕ плодим по decision на копию --
                # одна физическая цель на переднем крае, копии производны от неё.
                decisions.append(TargetDecision(
                    kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                    decision="target", reason="leading_edge", n_false=0,
                ))
                continue

            # v.kind == TARGET (одиночка либо нерегулярный fallback §4.12).
            if self._is_jitter_of_barrage(v, barrages):
                continue  # джиттер-осколок самозащитной помехи -- поглощён, не decision
            decisions.append(TargetDecision(
                kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                decision="target", reason="single", n_false=0,
            ))
        return decisions

    def _is_jitter_of_barrage(self, v: RangeVerdict, barrages: list[RangeVerdict]) -> bool:
        return any(
            abs(v.kx - b.kx) <= self._tol and abs(v.ky - b.ky) <= self._tol
            for b in barrages
        )


class CodeArbiter(Arbiter):
    """Вариант 2 (§5.3, задел): свежесть FM-m кода -- реализация после гл.6 (коррелятор).

    Абстракция заведена сейчас (LSP -- та же Strategy-форма, что `EdgeArbiter`),
    чтобы вызывающий код мог переключиться без изменений, когда появится
    многолучевой FM-m опрос. Само согласование с текущим кодом здесь ЕЩЁ не
    реализовано -- нужен коррелятор гл.6, которого пока нет.
    """

    def arbitrate(self, verdicts: list[RangeVerdict]) -> list[TargetDecision]:
        raise NotImplementedError("FM-m код-арбитр — гл.5.3/6, задел под опрос")
