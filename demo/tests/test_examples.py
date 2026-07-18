"""Приёмка примеров demo/ через `common.runner.TestRunner` (🚫 pytest, правило 04)."""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работала форма `python demo/tests/test_examples.py` (конвенция репо).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.generators.waveforms import SignalField
from demo.core import DemoContext
from demo.ex1_am_line.example import Ex1AmLine


class Ex1AmLineTests(TestRunner):

    def setup(self) -> None:
        self.ex = Ex1AmLine()
        self.ctx = DemoContext(name=self.ex.name, cfg=None, rng=np.random.default_rng(7))

    def test_name(self) -> AssertionGroup:
        g = AssertionGroup("ex1.name")
        g.add(self.ex.name == "ex1_am_line", f"name должен быть 'ex1_am_line', получено {self.ex.name!r}")
        return g

    def test_build_signal(self) -> AssertionGroup:
        g = AssertionGroup("ex1.build_signal")
        sig = self.ex.build_signal(self.ctx)
        g.add(isinstance(sig, SignalField), "build_signal должен вернуть SignalField")
        g.add(sig.data.ndim == 3, f"data.ndim должен быть 3, получено {sig.data.ndim}")
        g.add(sig.data.dtype == np.complex64, f"dtype должен быть complex64, получено {sig.data.dtype}")
        energy_in_window = float(np.sum(np.abs(sig.data) ** 2))
        g.add(energy_in_window > 0.0, "энергия сигнала в окне должна быть > 0")
        return g

    def test_am_carrier_peak(self) -> AssertionGroup:
        g = AssertionGroup("ex1.am_carrier_peak")
        sig = self.ex.build_signal(self.ctx)
        trace = sig.data[0, 0, :]
        spectrum = np.abs(np.fft.fft(trace))
        g.add(spectrum.size > 0 and float(np.sum(spectrum)) > 0.0, "спектр не должен быть пустым")

        freqs = np.fft.fftfreq(trace.size, d=1.0 / sig.fs)
        peak_freq = float(freqs[int(np.argmax(spectrum))])
        expected = 100e6
        tolerance = sig.fs / trace.size * 3  # несколько бинов допуска
        g.add(abs(abs(peak_freq) - expected) <= tolerance,
              f"пик спектра ({peak_freq/1e6:.2f} МГц) должен быть у несущей "
              f"{expected/1e6:.0f} МГц ±{tolerance/1e6:.2f} МГц")
        return g

    def test_report_no_save(self) -> AssertionGroup:
        g = AssertionGroup("ex1.report_no_save")
        rep = self.ex.run(save=False)
        g.add(rep.example == "ex1_am_line", f"example должен быть 'ex1_am_line', получено {rep.example!r}")
        g.add(rep.figures == [], "figures должен быть пуст при save=False")
        g.add(rep.metrics.get("n_samples") == 4096,
              f"metrics['n_samples'] должен быть 4096, получено {rep.metrics.get('n_samples')}")
        g.add(rep.metrics.get("m") == 0.5,
              f"metrics['m'] должен быть 0.5, получено {rep.metrics.get('m')}")
        return g


if __name__ == "__main__":
    Ex1AmLineTests().run_all()
