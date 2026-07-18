"""Приёмка ex1-matched: согласованный фильтр по месту СТФТ (🚫 pytest, правило 04)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from demo.ex1_am_line.denoise import match_counts, true_intervals  # noqa: E402
from demo.ex1_am_line.example import (  # noqa: E402
    FS,
    KIND_AM,
    SEED,
    _line_signal,
    add_noise_at_snr,
)
from demo.ex1_am_line.matched import MatchedPipeline, parabolic_refine_hz  # noqa: E402


class Ex1MatchedTests(TestRunner):

    def setup(self) -> None:
        self.pipe = MatchedPipeline()
        self.f_c = 100e6
        self.clean = _line_signal(KIND_AM, self.f_c)
        self.truth = true_intervals(KIND_AM, self.f_c)

    def test_clean_all_found(self) -> AssertionGroup:
        g = AssertionGroup("matched.clean")
        pulses = self.pipe.find(self.clean)
        found, false = match_counts(pulses, self.truth)
        g.add(found == 3, f"чистый: найдены все 3, получено {found}")
        g.add(false == 0, f"чистый: ложных 0, получено {false}")
        starts = sorted(p.start for p in pulses)
        for s, (a, _) in zip(starts, self.truth, strict=False):
            g.add(abs(s - a) <= 4, f"t0 уточнён согл.фильтром: |{s}-{a}| <= 4")
        for p in pulses:
            g.add(abs(abs(p.carrier_hz) - self.f_c) < 1e6,
                  f"f̂={p.carrier_hz / 1e6:+.2f} МГц в ±1 МГц от 100 (парабола)")
        return g

    def test_kind_am_on_long(self) -> AssertionGroup:
        g = AssertionGroup("matched.kind_am")
        pulses = self.pipe.find(self.clean)
        # тип проверяем на длительностях >=8 периодов: на 4п (20-40 отсч) am/radio
        # физически неразличимы (огибающая не успевает проявиться) -- честно не требуем.
        long_kinds = [p.kind for p in pulses if p.n_units >= 8]
        g.add(len(long_kinds) >= 2 and all(k == KIND_AM for k in long_kinds),
              f"длинные импульсы классифицированы am, получено {long_kinds}")
        return g

    def test_snr0_all_found(self) -> AssertionGroup:
        g = AssertionGroup("matched.snr0")
        noisy = add_noise_at_snr(self.clean, 0.0, np.random.default_rng(SEED))
        found, false = match_counts(self.pipe.find(noisy), self.truth)
        g.add(found == 3, f"SNR=0 дБ: найдены все 3 (порог ТЗ), получено {found}")
        g.add(false == 0, f"SNR=0 дБ: ложных 0, получено {false}")
        return g

    def test_snr_minus6_long_found(self) -> AssertionGroup:
        g = AssertionGroup("matched.snr-6_long")
        clean50 = _line_signal(KIND_AM, 50e6)
        truth50 = true_intervals(KIND_AM, 50e6)
        noisy = add_noise_at_snr(clean50, -6.0, np.random.default_rng(SEED))
        pulses = self.pipe.find(noisy)
        found, false = match_counts(pulses, truth50)
        g.add(found >= 1, f"SNR=-6: длинный (160 отсч) вытащен большим окном+СФ, найдено {found}")
        g.add(false == 0, f"SNR=-6: ложных 0, получено {false}")
        if pulses:
            g.add(any(abs(p.start - 2900) <= 16 for p in pulses),
                  "найденный на -6 дБ лежит у истинного t0=2900")
        return g

    def test_parabola_accuracy(self) -> AssertionGroup:
        g = AssertionGroup("matched.parabola")
        roi = self.clean[1600:1680]                     # 8 периодов несущей 100 МГц
        err = abs(abs(parabolic_refine_hz(roi, FS)) - self.f_c)
        g.add(err < 0.1e6, f"парабола 3 точек: |f̂-100 МГц| = {err / 1e3:.1f} кГц (< 100 кГц)")
        return g

    def test_input_not_mutated(self) -> AssertionGroup:
        g = AssertionGroup("matched.no_mutation")
        noisy = add_noise_at_snr(self.clean, 3.0, np.random.default_rng(SEED))
        backup = noisy.copy()
        self.pipe.find(noisy)
        g.add(bool(np.array_equal(noisy, backup)), "вход не мутирован")
        return g


if __name__ == "__main__":
    Ex1MatchedTests().run_all()
