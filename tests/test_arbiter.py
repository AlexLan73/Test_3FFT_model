"""Тесты физического арбитра гл.5 (`Doc/Patent/glava5_peredniy_krai.md`, передний край τ≥0).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_arbiter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.generators.waveforms.mseq import m_sequence_pow2  # noqa: E402
from core.models.tokenizer import (  # noqa: E402
    BARRAGE,
    COMB,
    TARGET,
    CodeArbiter,
    CombinedArbiter,
    EdgeArbiter,
    RangeVerdict,
    fm_correlate,
)


def _verdict(kx: float, ky: float, kind: str, lead_r: int, period_dr: float | None = None) -> RangeVerdict:
    return RangeVerdict(kx=kx, ky=ky, kind=kind, lead_r=lead_r, period_dr=period_dr)


class EdgeArbiterTests(TestRunner):

    def setup(self) -> None:
        self.arbiter = EdgeArbiter()

    def test_single_target_verdict_is_target(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.single_target_verdict_is_target")
        v = _verdict(kx=2.0, ky=-3.0, kind=TARGET, lead_r=5)
        decisions = self.arbiter.arbitrate([v])
        g.add(len(decisions) == 1, f"должно быть 1 decision, получено {len(decisions)}")
        if decisions:
            d = decisions[0]
            g.add(d.decision == "target", f"одиночка -> target, получено {d.decision}")
            g.add(d.reason == "single", f"reason должен быть 'single', получено {d.reason}")
            g.add(d.lead_r == 5, f"lead_r должен быть 5, получено {d.lead_r}")
        return g

    def test_comb_verdict_is_target_on_leading_edge(self) -> AssertionGroup:
        """Гребёнка ретранслятора (§5.2): ведущий (ближний по дальности) член -- истинная
        цель. Решение ПО ДАЛЬНОСТИ, не по яркости -- lead_r берётся из RangeVerdict как есть
        (min(r) сборки прохода 2, `assemble_range`), арбитр его не пересчитывает по амплитуде."""
        g = AssertionGroup("arbiter.comb_verdict_is_target_on_leading_edge")
        v = _verdict(kx=-6.0, ky=5.0, kind=COMB, lead_r=4, period_dr=4.0)
        decisions = self.arbiter.arbitrate([v])
        g.add(len(decisions) == 1, f"должен быть 1 decision (не по одному на копию), получено {len(decisions)}")
        if decisions:
            d = decisions[0]
            g.add(d.decision == "target", f"передний край гребёнки -> target, получено {d.decision}")
            g.add(d.reason == "leading_edge", f"reason должен быть 'leading_edge', получено {d.reason}")
            g.add(d.lead_r == 4, f"lead_r должен остаться ведущим (min r)=4, получено {d.lead_r}")
        return g

    def test_barrage_verdict_is_jammer(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.barrage_verdict_is_jammer")
        v = _verdict(kx=0.0, ky=0.0, kind=BARRAGE, lead_r=0)
        decisions = self.arbiter.arbitrate([v])
        g.add(len(decisions) == 1, f"должен быть 1 decision, получено {len(decisions)}")
        if decisions:
            d = decisions[0]
            g.add(d.decision == "jammer", f"заград -> jammer, получено {d.decision}")
            g.add(d.reason == "barrage_selfscreen", f"reason должен быть 'barrage_selfscreen', получено {d.reason}")
            g.add(d.lead_r == 0, f"lead_r должен сохраниться (кандидат-цель под самоприкрытием), получено {d.lead_r}")
        return g

    def test_jitter_targets_near_barrage_are_absorbed(self) -> AssertionGroup:
        """§5.5: заград джиттерит по углу -- мелкие 'target'-осколки под СОСЕДНИМИ (±1 бин)
        углами -- та же самозащитная помеха, не отдельные цели -- поглощаются, НЕ дают
        decision. Дальняя ОДИНОЧНАЯ цель (далеко от любого barrage) -- остаётся target."""
        g = AssertionGroup("arbiter.jitter_targets_near_barrage_are_absorbed")
        verdicts = [
            _verdict(kx=0.0, ky=0.0, kind=BARRAGE, lead_r=10),
            _verdict(kx=1.0, ky=0.0, kind=TARGET, lead_r=11),    # джиттер-осколок (|Δkx|=1)
            _verdict(kx=0.0, ky=-1.0, kind=TARGET, lead_r=9),    # джиттер-осколок (|Δky|=1)
            _verdict(kx=8.0, ky=8.0, kind=TARGET, lead_r=40),    # далеко от заграда -- настоящая цель
        ]
        decisions = self.arbiter.arbitrate(verdicts)

        g.add(len(decisions) == 2,
              f"должно остаться 2 decision (barrage + дальняя цель), получено {len(decisions)}: {decisions}")

        by_kind = {(d.kx, d.ky): d for d in decisions}
        g.add((0.0, 0.0) in by_kind, "заград (0,0) должен остаться в решениях")
        if (0.0, 0.0) in by_kind:
            g.add(by_kind[(0.0, 0.0)].decision == "jammer", "заград (0,0) должен быть jammer")

        g.add((8.0, 8.0) in by_kind, "дальняя цель (8,8) должна остаться в решениях")
        if (8.0, 8.0) in by_kind:
            d = by_kind[(8.0, 8.0)]
            g.add(d.decision == "target", f"дальняя цель -> target, получено {d.decision}")
            g.add(d.reason == "single", f"reason должен быть 'single', получено {d.reason}")

        g.add((1.0, 0.0) not in by_kind, "джиттер-осколок (1,0) должен быть поглощён, не decision")
        g.add((0.0, -1.0) not in by_kind, "джиттер-осколок (0,-1) должен быть поглощён, не decision")
        return g

    def test_angle_merge_tol_is_configurable(self) -> AssertionGroup:
        """`angle_merge_tol=0` -- строгое совпадение угла: соседний (±1) target НЕ поглощается."""
        g = AssertionGroup("arbiter.angle_merge_tol_is_configurable")
        strict_arbiter = EdgeArbiter(angle_merge_tol=0)
        verdicts = [
            _verdict(kx=0.0, ky=0.0, kind=BARRAGE, lead_r=10),
            _verdict(kx=1.0, ky=0.0, kind=TARGET, lead_r=11),
        ]
        decisions = strict_arbiter.arbitrate(verdicts)
        g.add(len(decisions) == 2,
              f"tol=0 -- сосед НЕ поглощается, должно быть 2 decision, получено {len(decisions)}")
        return g

    def test_does_not_mutate_input(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.does_not_mutate_input")
        verdicts = [_verdict(kx=2.0, ky=-3.0, kind=TARGET, lead_r=5)]
        snapshot = list(verdicts)
        _ = self.arbiter.arbitrate(verdicts)
        g.add(verdicts == snapshot, "arbitrate() не должен мутировать входной список verdicts")
        return g


class CodeArbiterTests(TestRunner):
    """Вариант 2 (§5.3): свежесть FM-m кода через `fm_correlate` (§6.2)."""

    def setup(self) -> None:
        self.ref = m_sequence_pow2(degree=10, seed=1)          # свежий код такта
        # чужой код ретранслятора: НЕ сдвиг текущего (разный seed = та же M-послед. со
        # сдвигом фазы -> коррелирует!), а некоррелирующий ±1 код другого такта/полинома.
        self.other = np.sign(np.random.default_rng(7).standard_normal(self.ref.size)).astype(np.float32)
        self.noise = np.random.default_rng(0).standard_normal(self.ref.size).astype(np.float32)

    def test_fm_correlate_autocorr_sharp_peak(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.fm_correlate_autocorr")
        auto = fm_correlate(self.ref, self.ref)
        cross = fm_correlate(self.ref, self.other)
        g.add(int(np.argmax(auto)) == 0, "автокорреляция -- пик на нулевой задержке")
        g.add(float(auto.max()) > 5.0 * float(cross.max()),
              f"пик своего кода >> чужого: auto={auto.max():.3g} cross={cross.max():.3g}")
        return g

    def test_fresh_code_is_target(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.code_fresh_target")
        arb = CodeArbiter(ref_code=self.ref, signal_by_angle={(0.0, 0.0): self.ref})
        d = arb.arbitrate([_verdict(0.0, 0.0, TARGET, lead_r=5)])[0]
        g.add(d.decision == "target" and d.reason == "fresh_code",
              f"свежий код -> target/fresh_code, получено {d.decision}/{d.reason}")
        return g

    def test_stale_code_is_false(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.code_stale_false")
        arb = CodeArbiter(ref_code=self.ref, signal_by_angle={(0.0, 0.0): self.other})
        d = arb.arbitrate([_verdict(0.0, 0.0, TARGET, lead_r=5)])[0]
        g.add(d.decision == "false" and d.reason == "stale_code",
              f"чужой код (ретранслятор) -> false/stale_code, получено {d.decision}/{d.reason}")
        return g

    def test_noise_is_false(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.code_noise_false")
        arb = CodeArbiter(ref_code=self.ref, signal_by_angle={(0.0, 0.0): self.noise})
        d = arb.arbitrate([_verdict(0.0, 0.0, TARGET, lead_r=5)])[0]
        g.add(d.decision == "false", f"шумовой осколок -> false, получено {d.decision}")
        return g

    def test_no_probe_stays_candidate(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.code_no_probe")
        arb = CodeArbiter(ref_code=self.ref, signal_by_angle={})  # угол не опрашивался
        d = arb.arbitrate([_verdict(0.0, 0.0, TARGET, lead_r=5)])[0]
        g.add(d.decision == "target" and d.reason == "no_probe",
              "неопрошенный угол -> кандидат (§4.8, передний край терять нельзя)")
        return g

    def test_barrage_stays_jammer(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.code_barrage_jammer")
        arb = CodeArbiter(ref_code=self.ref, signal_by_angle={})
        d = arb.arbitrate([_verdict(0.0, 0.0, BARRAGE, lead_r=0)])[0]
        g.add(d.decision == "jammer", "заград кодом не опрашивается -> jammer")
        return g


class CombinedArbiterTests(TestRunner):
    """§5.4: геометрия (EdgeArbiter) И код (CodeArbiter) -- закрывают слабости друг друга."""

    def setup(self) -> None:
        self.ref = m_sequence_pow2(degree=10, seed=1)
        self.other = np.sign(np.random.default_rng(7).standard_normal(self.ref.size)).astype(np.float32)

    def test_comb_leading_edge_target_without_code(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.combined_comb_target")
        # гребёнка -> сильная геометрия (передний край), код не нужен
        toks = [_verdict(2.0, 3.0, COMB, lead_r=10, period_dr=10.0)]
        code = CodeArbiter(ref_code=self.ref, signal_by_angle={})   # даже без опроса
        d = CombinedArbiter(EdgeArbiter(), code).arbitrate(toks)[0]
        g.add(d.decision == "target" and d.reason == "leading_edge",
              f"comb -> target геометрией (§5.4), получено {d.decision}/{d.reason}")
        return g

    def test_single_stale_code_cleared(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.combined_single_cleared")
        # одиночка (слабая геометрия) + чужой код -> ЧИСТИТСЯ Вариантом 2
        v = _verdict(1.0, 1.0, TARGET, lead_r=2)
        code = CodeArbiter(ref_code=self.ref, signal_by_angle={(1.0, 1.0): self.other})
        decisions = CombinedArbiter(EdgeArbiter(), code).arbitrate([v])
        d = decisions[0]
        g.add(d.decision == "false" and d.reason == "cleared_by_code",
              f"одиночка+чужой код -> false/cleared_by_code (чистка осколка), получено {d.decision}/{d.reason}")
        return g

    def test_single_fresh_code_target(self) -> AssertionGroup:
        g = AssertionGroup("arbiter.combined_single_fresh")
        v = _verdict(1.0, 1.0, TARGET, lead_r=2)
        code = CodeArbiter(ref_code=self.ref, signal_by_angle={(1.0, 1.0): self.ref})
        d = CombinedArbiter(EdgeArbiter(), code).arbitrate([v])[0]
        g.add(d.decision == "target" and d.reason == "edge_and_code",
              f"одиночка+свежий код -> target/edge_and_code, получено {d.decision}/{d.reason}")
        return g


if __name__ == "__main__":
    ok = True
    for cls in (EdgeArbiterTests, CodeArbiterTests, CombinedArbiterTests):
        ok = cls().run_all() and ok
    sys.exit(0 if ok else 1)
