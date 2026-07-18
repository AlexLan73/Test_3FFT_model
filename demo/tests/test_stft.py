"""Приёмка ex1-stft: детекция по спектрограмме (🚫 pytest, правило 04)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from demo.ex1_am_line.denoise import match_counts, true_intervals  # noqa: E402
from demo.ex1_am_line.example import KIND_AM, SEED, _line_signal, add_noise_at_snr  # noqa: E402
from demo.ex1_am_line.stft_detect import StftDetector, StftParams, stft_power  # noqa: E402


class Ex1StftTests(TestRunner):

    def setup(self) -> None:
        self.det = StftDetector()
        self.kind, self.f_c = KIND_AM, 100e6
        self.clean = _line_signal(self.kind, self.f_c)
        self.truth = true_intervals(self.kind, self.f_c)

    def test_stft_shape(self) -> AssertionGroup:
        g = AssertionGroup("stft.shape")
        prm = StftParams()
        s = stft_power(self.clean, prm)
        n_frames = 1 + (len(self.clean) - prm.win_len) // prm.hop
        g.add(s.shape == (n_frames, 32), f"форма S {(n_frames, 32)}, получено {s.shape}")
        g.add(prm.n_fft == 32, f"n_fft = 16+16 = 32, получено {prm.n_fft}")
        g.add(bool(np.all(s >= 0.0)), "мощность неотрицательна")
        return g

    def test_clean_all_found(self) -> AssertionGroup:
        g = AssertionGroup("stft.clean_all_found")
        found, false = match_counts(self.det.detect(self.clean), self.truth)
        g.add(found == 3, f"чистый: найдены все 3, получено {found}")
        g.add(false == 0, f"чистый: ложных 0, получено {false}")
        return g

    def test_snr10_found(self) -> AssertionGroup:
        g = AssertionGroup("stft.snr10_found")
        noisy = add_noise_at_snr(self.clean, 10.0, np.random.default_rng(SEED))
        found, false = match_counts(self.det.detect(noisy), self.truth)
        g.add(found == 3, f"SNR=+10: найдены все 3, получено {found}")
        g.add(false == 0, f"SNR=+10: ложных 0, получено {false}")
        return g

    def test_carrier_estimate(self) -> AssertionGroup:
        g = AssertionGroup("stft.carrier_estimate")
        f_hat = self.det.carrier_hz_est(self.clean)
        # разрешение СТФТ-32 = fs/32 = 15.6 МГц -> допуск 1 бин
        g.add(abs(abs(f_hat) - self.f_c) <= FS_BIN,
              f"|f̂|={abs(f_hat) / 1e6:.1f} МГц около 100 (±{FS_BIN / 1e6:.1f})")
        return g

    def test_no_signal_no_detect(self) -> AssertionGroup:
        g = AssertionGroup("stft.no_signal")
        noise_only = add_noise_at_snr(np.zeros_like(self.clean), 0.0,
                                      np.random.default_rng(SEED))
        n_pulses = len(self.det.detect(noise_only))
        g.add(n_pulses <= 1, f"чистый шум: сегментов <=1, получено {n_pulses}")
        return g

    def test_input_not_mutated(self) -> AssertionGroup:
        g = AssertionGroup("stft.no_mutation")
        noisy = add_noise_at_snr(self.clean, 3.0, np.random.default_rng(SEED))
        backup = noisy.copy()
        self.det.detect(noisy)
        g.add(bool(np.array_equal(noisy, backup)), "вход не мутирован")
        return g


FS_BIN = 500e6 / 32   # ширина бина FFT-32


if __name__ == "__main__":
    Ex1StftTests().run_all()
