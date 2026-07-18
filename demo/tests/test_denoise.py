"""Приёмка ex1-denoise: слепая детекция в белом шуме (🚫 pytest, правило 04)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from demo.ex1_am_line.denoise import (  # noqa: E402
    PulseDetector,
    SpectralGateFilter,
    WienerFilter,
    match_counts,
    run_denoise,
    true_intervals,
)
from demo.ex1_am_line.example import KIND_AM, SEED, _line_signal, add_noise_at_snr  # noqa: E402


class Ex1DenoiseTests(TestRunner):

    def setup(self) -> None:
        self.filters = (SpectralGateFilter(), WienerFilter())
        self.det = PulseDetector()
        self.kind, self.f_c = KIND_AM, 100e6
        self.clean = _line_signal(self.kind, self.f_c)
        self.truth = true_intervals(self.kind, self.f_c)

    def test_clean_all_found(self) -> AssertionGroup:
        g = AssertionGroup("denoise.clean_all_found")
        for flt in self.filters:
            res = run_denoise(self.clean, flt, self.det)
            found, false = match_counts(res.pulses, self.truth)
            g.add(found == 3, f"{flt.name}: чистый сигнал — найдены все 3, получено {found}")
            g.add(false == 0, f"{flt.name}: чистый сигнал — ложных 0, получено {false}")
        return g

    def test_snr10_found(self) -> AssertionGroup:
        g = AssertionGroup("denoise.snr10_found")
        noisy = add_noise_at_snr(self.clean, 10.0, np.random.default_rng(SEED))
        for flt in self.filters:
            res = run_denoise(noisy, flt, self.det)
            found, false = match_counts(res.pulses, self.truth)
            g.add(found == 3, f"{flt.name}: SNR=+10 — найдены все 3, получено {found}")
            g.add(false == 0, f"{flt.name}: SNR=+10 — ложных 0, получено {false}")
        return g

    def test_carrier_estimate(self) -> AssertionGroup:
        g = AssertionGroup("denoise.carrier_estimate")
        for flt in self.filters:
            res = run_denoise(self.clean, flt, self.det)
            err = abs(abs(res.carrier_hz_est) - self.f_c)
            g.add(err < 2e6, f"{flt.name}: |f̂|-100 МГц = {err / 1e6:.2f} МГц (< 2 МГц)")
        return g

    def test_no_signal_no_detect(self) -> AssertionGroup:
        g = AssertionGroup("denoise.no_signal")
        noise_only = add_noise_at_snr(np.zeros_like(self.clean), 0.0,
                                      np.random.default_rng(SEED))
        for flt in self.filters:
            res = run_denoise(noise_only, flt, self.det)
            # Pfa=1e-3 на 4096 ячеек -> ~4 одиночных срабатывания, но min_len=3 их режет:
            # допускаем <=1 случайный сегмент (детерминированный seed).
            g.add(len(res.pulses) <= 1,
                  f"{flt.name}: чистый шум — сегментов <=1, получено {len(res.pulses)}")
        return g

    def test_filters_do_not_mutate(self) -> AssertionGroup:
        g = AssertionGroup("denoise.no_mutation")
        noisy = add_noise_at_snr(self.clean, 3.0, np.random.default_rng(SEED))
        backup = noisy.copy()
        for flt in self.filters:
            flt.apply(noisy)
            g.add(bool(np.array_equal(noisy, backup)), f"{flt.name}: вход не мутирован")
        return g


if __name__ == "__main__":
    Ex1DenoiseTests().run_all()
