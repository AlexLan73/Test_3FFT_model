"""Тесты SNR-эстиматора (БЕЗ pytest — только TestRunner).

Запуск:  python tests/test_snr.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.snr import (  # noqa: E402
    PointSignalGenerator,
    SnrConfig,
    SpectrumSnrEstimator,
    StatisticsSnrEstimator,
    compute_pipeline_sizes,
)

_N = 2048
_FREQ = 0.15


class SnrTests(TestRunner):

    def setup(self) -> None:
        self.gen = PointSignalGenerator()
        self.spec = SpectrumSnrEstimator(SnrConfig())
        self.stat = StatisticsSnrEstimator()

    # ── 1. Генератор: структура строба (на чистом тоне, noise_power=0) ────────
    def test_generator_strobe(self) -> AssertionGroup:
        g = AssertionGroup("snr.generator_strobe")
        frac = 0.25
        # NB: A=√(σ²·10^(SNR/10)) → при noise_power=0 тон исчезает; тестируем с шумом.
        sig, sup = self.gen.generate(_N, _FREQ, 20.0, frac, "left", 1.0,
                                     np.random.default_rng(0))
        g.add(sig.shape == (_N,), f"длина сигнала {_N}, получено {sig.shape}")
        g.add(sig.dtype == np.complex64, f"dtype complex64, получено {sig.dtype}")
        g.add(sup.start == 0, f"left → строб с 0, получено start={sup.start}")
        exp_len = round(frac * _N)
        g.add(abs((sup.stop - sup.start) - exp_len) <= 1,
              f"длина строба ≈ {exp_len}, получено {sup.stop - sup.start}")
        p_in = float((np.abs(sig[sup]) ** 2).mean())
        p_out = float((np.abs(sig[sup.stop:]) ** 2).mean())
        g.add(p_in > 5.0 * p_out, f"мощность в стробе ({p_in:.2f}) >> вне ({p_out:.2f})")
        return g

    # ── 2. Позиция right → строб в конце ──────────────────────────────────────
    def test_generator_position_right(self) -> AssertionGroup:
        g = AssertionGroup("snr.generator_position_right")
        frac = 0.1
        sig, sup = self.gen.generate(_N, _FREQ, 20.0, frac, "right", 1.0,
                                     np.random.default_rng(1))
        g.add(sup.stop == _N, f"right → строб до конца, получено stop={sup.stop}")
        p_in = float((np.abs(sig[sup]) ** 2).mean())
        p_before = float((np.abs(sig[:sup.start]) ** 2).mean())
        g.add(p_in > 5.0 * p_before,
              f"мощность в стробе-хвосте ({p_in:.2f}) >> до него ({p_before:.2f})")
        return g

    # ── 3. Спектр: монотонный рост с SNR_in ───────────────────────────────────
    def test_spectrum_monotonic(self) -> AssertionGroup:
        g = AssertionGroup("snr.spectrum_monotonic")
        means = []
        for snr_in in (0.0, 20.0, 40.0):
            vals = [self.spec.estimate(
                        self.gen.generate(_N, _FREQ, snr_in, 1.0, "center", 1.0,
                                          np.random.default_rng(s))[0]).snr_db
                    for s in range(20)]
            means.append(float(np.mean(vals)))
        g.add(means[0] < means[1] < means[2],
              f"спектр должен расти с SNR_in: {[round(m, 1) for m in means]}")
        return g

    # ── 4. Спектр на чистом тоне: пик в нужном бине + теория ───────────────────
    def test_spectrum_clean_tone(self) -> AssertionGroup:
        g = AssertionGroup("snr.spectrum_clean_tone")
        _, n_actual, n_fft = compute_pipeline_sizes(_N, 2048, 0)
        sig, _sup = self.gen.generate(_N, _FREQ, 20.0, 1.0, "center", 1.0,
                                      np.random.default_rng(3))
        res = self.spec.estimate(sig)
        k_expect = round(_FREQ * n_fft)
        g.add(res.k_peak is not None and abs(res.k_peak - k_expect) <= 1,
              f"пик ожидается ~{k_expect}, получено {res.k_peak}")
        theory = 20.0 + 10.0 * math.log10(n_actual)
        g.add(abs(res.snr_db - theory) <= 4.0,
              f"SNR_fft={res.snr_db:.1f} ≈ теория {theory:.1f} ±4 дБ (Hann loss + CFAR bias)")
        return g

    # ── 5. Статистика на высоком SNR (frac=0.5, не 1.0 — иначе вырождается) ────
    def test_statistics_high_snr(self) -> AssertionGroup:
        g = AssertionGroup("snr.statistics_high_snr")
        vals = []
        for s in range(20):
            sig, sup = self.gen.generate(_N, _FREQ, 30.0, 0.5, "center", 1.0,
                                         np.random.default_rng(s))
            vals.append(self.stat.estimate(sig, sup).snr_db)
        mean = float(np.mean(vals))
        g.add(abs(mean - 30.0) < 1.0,
              f"статистика @30дБ должна быть ≈30 (нет processing gain), получено {mean:.2f}")
        return g

    # ── 6. Статистика требует support ─────────────────────────────────────────
    def test_statistics_requires_support(self) -> AssertionGroup:
        g = AssertionGroup("snr.statistics_requires_support")
        sig, _sup = self.gen.generate(_N, _FREQ, 10.0, 0.5, "center", 1.0,
                                      np.random.default_rng(0))
        raised = False
        try:
            self.stat.estimate(sig, None)
        except ValueError:
            raised = True
        g.add(raised, "estimate(signal, None) должен кинуть ValueError")
        return g

    # ── 7. H0 (чистый шум): обе оценки ограничены и конечны ───────────────────
    def test_h0_noise(self) -> AssertionGroup:
        g = AssertionGroup("snr.h0_noise")
        # чистый шум: SNR_in очень низкий → тон пренебрежимо мал
        sig, sup = self.gen.generate(_N, _FREQ, -100.0, 0.5, "center", 1.0,
                                     np.random.default_rng(11))
        st = self.stat.estimate(sig, sup).snr_db
        g.add(math.isfinite(st) and st < 3.0,
              f"статистика H0 должна быть <3 дБ и конечна, получено {st:.1f}")
        # спектр H0 vs спектр при SNR_in=30
        sp_h0 = self.spec.estimate(sig).snr_db
        sig30, _ = self.gen.generate(_N, _FREQ, 30.0, 1.0, "center", 1.0,
                                     np.random.default_rng(11))
        sp_30 = self.spec.estimate(sig30).snr_db
        g.add(sp_h0 < 0.5 * sp_30,
              f"спектр H0 ({sp_h0:.1f}) должен быть << спектра@30дБ ({sp_30:.1f})")
        return g


if __name__ == "__main__":
    SnrTests().run_all()
