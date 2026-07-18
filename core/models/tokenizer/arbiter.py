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
свежесть FM-m кода, гл.6 §6.2 коррелятор) -- принятый сигнал сжимается в
острый пик своим (свежим) кодом такта; ретранслятор несёт чужой/старый код --
пик не формируется (проигрыш согласования ~10·log10(L) дБ). `CombinedArbiter`
-- Composite §5.4: геометрия и код закрывают слабые места друг друга.

Область действия (§5.5): правило `barrage_selfscreen` строго для
САМОЗАЩИТНОЙ помехи (носитель = цель, тот же угол) -- джиттер-осколки
`target`-вердиктов под соседними углами (пик заграда "уезжает" в соседний бин,
`MemoryBank/specs/tokenizer_barrage_pass2_2026-07-17.md`) поглощаются заградом
угловой толерантностью, а не порождают ложные target-decision'ы.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

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


def fm_correlate(ref: np.ndarray, inp: np.ndarray) -> np.ndarray:
    """Корреляция через БПФ (гл.6 §6.2): `IFFT(conj(FFT(ref))·FFT(inp))`.

    Размер преобразования -- ближайшая степень двойки `>= max(len(ref), len(inp))`
    (§6.2: "размер преобразования — под длину кода такта, дополнение нулями до
    2ⁿ"). Возвращает магнитуду `|·|/N` (реальный массив длины `N`) -- пик в точке
    `j` означает совпадение кода с задержкой `j`. Не мутирует входы (`np.fft.*`
    возвращает новые массивы).
    """
    n = max(len(ref), len(inp))
    n_fft = 1 << (n - 1).bit_length()  # ближайшая степень двойки >= n
    ref_spec = np.fft.fft(ref, n=n_fft)
    inp_spec = np.fft.fft(inp, n=n_fft)
    corr = np.fft.ifft(np.conj(ref_spec) * inp_spec)
    return np.abs(corr) / n_fft


def _peak_to_floor_db(corr: np.ndarray) -> float:
    """Пик/фон в дБ: `20·log10(max(corr) / rms(corr))`.

    RMS (`sqrt(mean(corr**2))`) по ВСЕМУ массиву -- пик даёт лишь один отсчёт
    из `N` и почти не искажает оценку фона, а RMS устойчивее `median()` к
    статистике экстремумов случайного шума (эмпирически, degree=10..13:
    медиана как фон завышает пик/фон для случайного/чужого кода до ~13-15 дБ
    только за счёт statистики максимума среди N отсчётов -- ложно похоже на
    "код найден"; RMS даёт для шума/чужого кода стабильно ~11-12 дБ против
    ~29-38 дБ для настоящего совпадения -- разделение с запасом).
    """
    peak = float(np.max(corr))
    rms = float(np.sqrt(np.mean(corr**2)))
    eps = 1e-12
    return 20.0 * np.log10((peak + eps) / (rms + eps))


class CodeArbiter(Arbiter):
    """Вариант 2 (§5.3, §6.2): свежесть FM-m кода -- реализован поверх `fm_correlate`.

    Истинная цель переизлучает СВЕЖИЙ код такта -> корреляция с текущим
    эталонным кодом (`ref_code`) сжимается в острый пик (§6.2). Ретранслятор
    несёт НЕ ТОТ код (старый/чужой) -> корреляция не сжимается, пик не
    формируется (проигрыш согласования ~10·log10(L) дБ, §5.3).

    Свежесть требует ПРИНЯТОГО сигнала по каждому кандидату -- инжектируется
    через DI (`signal_by_angle`), а не вычисляется здесь (коррелятор гл.6 --
    отдельный, GPU-ускоренный движок; здесь -- решающее правило поверх его
    результата, в чистом numpy для арбитра-заглушки/офлайн-проверки).

    Parameters
    ----------
    ref_code : текущий свежий код такта (напр. `m_sequence_pow2(...)`), ±1 или комплексный.
    signal_by_angle : принятый сигнал луча по углу `(kx, ky)` -> сырой сигнал (комплексный/±1).
    peak_threshold_db : порог пик/фон (§6.9 "превышение пика над фоном") для "код найден".
        Дефолт **20.0 дБ** (а не 6.0) -- калибровано эмпирически по
        `_peak_to_floor_db` (RMS-фон): совпадение свежего кода -- ~29-38 дБ
        (degree=10..13), шум/чужой код (другой полином той же степени) --
        стабильно ~11-12 дБ; 6 дБ не разделял бы шум и цель, 20 дБ -- с запасом.
    """

    def __init__(
        self,
        ref_code: np.ndarray | None = None,
        signal_by_angle: dict[tuple[float, float], np.ndarray] | None = None,
        peak_threshold_db: float = 20.0,
    ) -> None:
        self._ref_code = ref_code
        self._signal_by_angle = signal_by_angle if signal_by_angle is not None else {}
        self._peak_threshold_db = peak_threshold_db

    def arbitrate(self, verdicts: list[RangeVerdict]) -> list[TargetDecision]:
        decisions: list[TargetDecision] = []
        for v in verdicts:
            if v.kind == BARRAGE:
                # §5.3: заград не опрашивается кодом -- решение как у EdgeArbiter.
                decisions.append(TargetDecision(
                    kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                    decision="jammer", reason="barrage_selfscreen", n_false=0,
                ))
                continue

            signal = self._signal_by_angle.get((v.kx, v.ky))
            if signal is None or self._ref_code is None:
                # код такта не опрашивался этот угол -- консервативно НЕ отбрасываем
                # кандидата (§4.8: передний край терять нельзя без улики против).
                decisions.append(TargetDecision(
                    kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                    decision="target", reason="no_probe", n_false=0,
                ))
                continue

            corr = fm_correlate(self._ref_code, signal)
            peak_db = _peak_to_floor_db(corr)
            if peak_db >= self._peak_threshold_db:
                decisions.append(TargetDecision(
                    kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                    decision="target", reason="fresh_code", n_false=0,
                ))
            else:
                decisions.append(TargetDecision(
                    kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                    decision="false", reason="stale_code", n_false=0,
                ))
        return decisions


class CombinedArbiter(Arbiter):
    """§5.4 (Composite): геометрия (`EdgeArbiter`) И код (`CodeArbiter`) -- закрывают
    слабые места друг друга. Не заменяет ни один из вариантов, комбинирует решения.

    Логика (не усложняем сверх §5.4):

    - `barrage` -- решение `EdgeArbiter` авторитетно (`jammer`), код заград не опрашивает.
    - причинностная группа с ведущим краем (`comb`, reason `leading_edge`) --
      **сильная** геометрическая улика (причинность τ≥0 доказывает передний
      край цепочки ретрансляции сама по себе) -- код не нужен, остаётся `target`.
    - одиночный кандидат без причинностной группы (reason `single`) --
      геометрия **слабая** (одна копия ничего не доказывает о τ≥0), требует
      подтверждения кодом:
        * код свежий (`fresh_code`) -> `target`/`edge_and_code`;
        * код не опрашивался (`no_probe`) -> консервативно `target`/`single_no_probe`
          (нет улики против, §4.8);
        * код несвежий (`stale_code`) -> **`false`/`cleared_by_code`** -- именно
          так Вариант 2 чистит осколки, которые геометрия Варианта 1 одна не
          отличает от настоящей одиночной цели.
    - кандидат, поглощённый `EdgeArbiter` как джиттер-осколок заграда (нет
      decision у edge) -- если код всё же свежий (истинная цель случайно легла
      рядом с заградом по углу), код "спасает" его -> `target`/`code_only`;
      иначе остаётся поглощённым (нет decision), как у `EdgeArbiter`.
    """

    def __init__(self, edge: EdgeArbiter, code: CodeArbiter) -> None:
        self._edge = edge
        self._code = code

    def arbitrate(self, verdicts: list[RangeVerdict]) -> list[TargetDecision]:
        edge_by_key = {(d.kx, d.ky): d for d in self._edge.arbitrate(verdicts)}
        code_by_key = {(d.kx, d.ky): d for d in self._code.arbitrate(verdicts)}

        decisions: list[TargetDecision] = []
        for v in verdicts:
            key = (v.kx, v.ky)

            if v.kind == BARRAGE:
                decisions.append(edge_by_key[key])
                continue

            edge_d = edge_by_key.get(key)
            code_d = code_by_key.get(key)
            code_fresh = (
                code_d is not None and code_d.decision == "target" and code_d.reason == "fresh_code"
            )

            if edge_d is None:
                # поглощён EdgeArbiter как джиттер-осколок заграда.
                if code_fresh:
                    decisions.append(TargetDecision(
                        kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                        decision="target", reason="code_only", n_false=0,
                    ))
                continue  # иначе остаётся поглощённым -- нет decision

            if edge_d.reason == "leading_edge":
                decisions.append(edge_d)
                continue

            # edge_d.reason == "single" -- слабая геометрия, нужна проверка кодом.
            if code_fresh:
                decisions.append(TargetDecision(
                    kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                    decision="target", reason="edge_and_code", n_false=0,
                ))
            elif code_d is not None and code_d.reason == "no_probe":
                decisions.append(TargetDecision(
                    kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                    decision="target", reason="single_no_probe", n_false=0,
                ))
            else:
                decisions.append(TargetDecision(
                    kx=v.kx, ky=v.ky, lead_r=v.lead_r,
                    decision="false", reason="cleared_by_code", n_false=0,
                ))
        return decisions
