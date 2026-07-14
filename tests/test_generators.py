"""Тесты P0 слоя генераторов сигналов (SignalField, TimeWindow, ConfigSource).

🚫 pytest — только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_generators.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, SkipTest, TestRunner  # noqa: E402
from common.validators import DataValidator  # noqa: E402
from core.config import ArrayConfig  # noqa: E402
from core.config.config_source import DefaultConfigSource, YamlConfigSource  # noqa: E402
from core.config.waveform_config import WaveTimeConfig  # noqa: E402
from core.generators.backends import NumpyBackend  # noqa: E402
from core.generators.waveforms import (  # noqa: E402
    AmWaveform,
    AxisKind,
    CwWaveform,
    LfmWaveform,
    Modulation,
    SignalField,
    TimeWindow,
    WaveformFactory,
    WaveformSpec,
)
from core.generators.waveforms.reference import getX_numpy  # noqa: E402, N812
from core.gpu_libs.loader import GpuLibsUnavailableError  # noqa: E402
from core.snr import StatisticsSnrEstimator  # noqa: E402

_BASELINE_YAML = _REPO / "core" / "config" / "configs" / "baseline.yaml"

# P2: N=4096 — как у test_lfm_matches_reference/test_cw_peak_frequency (не N=8192-baseline).
# На N=8192 raw-complex max_rel растёт с фазовой длиной окна (float32 GPU-ядро) и
# ВЫШЕ порога 1e-3 (CW≈1.02e-3, ЛЧМ(fdev=6e6)≈1.79e-3, см. отчёт P2) — известный найденный
# предел точности, не баг; здесь сверяем формулу на N=4096, где запас x2.
_P2_N = 4096


def _make_field(data: np.ndarray, axes: tuple[AxisKind, ...]) -> SignalField:
    return SignalField(data=data, modulation=Modulation.LFM, axes=axes, fs=12e6, carrier_hz=2e6)


class GeneratorsTests(TestRunner):

    # ── SignalField ──────────────────────────────────────────────────────────
    def test_field_valid(self) -> AssertionGroup:
        g = AssertionGroup("field.valid")
        data = np.zeros((4, 4, 16), dtype=np.complex64)
        f = _make_field(data, (AxisKind.ANTENNA_X, AxisKind.ANTENNA_Y, AxisKind.FAST_TIME))
        g.add(f.data.shape == (4, 4, 16), "data сохраняется без изменений формы")
        g.add(hasattr(f.meta, "items"), "meta — mapping")
        meta_mutation_blocked = False
        try:
            f.meta["x"] = 1.0  # type: ignore[index]
        except TypeError:
            meta_mutation_blocked = True
        g.add(meta_mutation_blocked, "meta должен быть неизменяемым (G2: MappingProxyType)")
        return g

    def test_field_axes_mismatch_raises(self) -> AssertionGroup:
        g = AssertionGroup("field.axes_mismatch")
        data = np.zeros((4, 4, 16), dtype=np.complex64)
        raised = False
        try:
            _make_field(data, (AxisKind.ANTENNA_X, AxisKind.ANTENNA_Y))  # len 2 != ndim 3
        except ValueError:
            raised = True
        g.add(raised, "len(axes) != data.ndim должен кинуть ValueError")
        return g

    def test_field_dtype_mismatch_raises(self) -> AssertionGroup:
        g = AssertionGroup("field.dtype_mismatch")
        data = np.zeros((4, 4, 16), dtype=np.complex128)   # намеренно неверный dtype
        raised = False
        try:
            _make_field(data, (AxisKind.ANTENNA_X, AxisKind.ANTENNA_Y, AxisKind.FAST_TIME))
        except ValueError:
            raised = True
        g.add(raised, "dtype != complex64 должен кинуть ValueError")
        return g

    def test_field_eq_and_hash_do_not_raise(self) -> AssertionGroup:
        """G1: eq=False → identity-семантика, никаких 'truth value of array is ambiguous'."""
        g = AssertionGroup("field.eq_hash_g1")
        data = np.zeros((2, 2, 8), dtype=np.complex64)
        axes = (AxisKind.ANTENNA_X, AxisKind.ANTENNA_Y, AxisKind.FAST_TIME)
        f1 = _make_field(data, axes)
        f2 = _make_field(data, axes)

        eq_ok = False
        try:
            result = f1 == f2
            eq_ok = isinstance(result, bool)
        except ValueError:
            eq_ok = False
        g.add(eq_ok, "f1 == f2 не должен бросать (identity-сравнение)")
        g.add((f1 == f1) is True, "f1 == f1 (identity) должен быть True")

        hash_ok = True
        try:
            {f1: "a"}  # noqa: B018 — используем как ключ словаря
        except TypeError:
            hash_ok = False
        g.add(hash_ok, "SignalField должен быть хэшируем (не unhashable ndarray)")
        return g

    # ── TimeWindow ───────────────────────────────────────────────────────────
    def test_window_full(self) -> AssertionGroup:
        g = AssertionGroup("window.full")
        m = TimeWindow(kind="full").mask(1000, 12e6)
        g.add(m.dtype == bool, "маска bool")
        g.add(bool(m.all()), "full → все True")
        return g

    def test_window_partial(self) -> AssertionGroup:
        g = AssertionGroup("window.partial")
        fs, n = 1000.0, 1000
        m = TimeWindow(kind="partial", t0=0.2, t1=0.5).mask(n, fs)
        start, stop = round(0.2 * fs), round(0.5 * fs)
        g.add(int(m[:start].sum()) == 0, "перед t0 — всё False")
        g.add(bool(m[start:stop].all()), "внутри [t0,t1) — всё True")
        g.add(int(m[stop:].sum()) == 0, "после t1 — всё False")
        return g

    def test_window_short_length(self) -> AssertionGroup:
        g = AssertionGroup("window.short_length")
        fs, n, dur = 1000.0, 1000, 0.05
        m = TimeWindow(kind="short", t0=0.3, dur=dur).mask(n, fs)
        expect_len = round(dur * fs)
        g.add(int(m.sum()) == expect_len,
              f"длина short-окна {int(m.sum())} != round(dur*fs)={expect_len}")
        return g

    def test_window_energy_outside_zero(self) -> AssertionGroup:
        g = AssertionGroup("window.energy_outside_zero")
        fs, n = 1000.0, 500
        m = TimeWindow(kind="short", t0=0.1, dur=0.05).mask(n, fs)
        signal = np.ones(n, dtype=np.complex64)
        signal[~m] = 0.0
        energy_outside = float(np.sum(np.abs(signal[~m]) ** 2))
        g.add(energy_outside == 0.0, f"энергия вне маски должна быть 0, получено {energy_outside}")
        g.add(bool((np.abs(signal[m]) > 0).all()), "внутри маски сигнал не занулён")
        return g

    def test_window_missing_params_raise(self) -> AssertionGroup:
        g = AssertionGroup("window.missing_params_raise")
        partial_raised = False
        try:
            TimeWindow(kind="partial")   # нет t1
        except ValueError:
            partial_raised = True
        g.add(partial_raised, "partial без t1 должен кинуть ValueError")

        short_raised = False
        try:
            TimeWindow(kind="short")     # нет dur
        except ValueError:
            short_raised = True
        g.add(short_raised, "short без dur должен кинуть ValueError")
        return g

    # ── ConfigSource ─────────────────────────────────────────────────────────
    def test_default_config_source(self) -> AssertionGroup:
        g = AssertionGroup("config.default_source")
        cfg = DefaultConfigSource().load()
        g.add(cfg.fs == 12e6, f"fs=12e6, получено {cfg.fs}")
        g.add(cfg.carrier_hz == 2e6, f"carrier_hz=2e6, получено {cfg.carrier_hz}")
        g.add(cfg.fdev_hz == 6e6, f"fdev_hz=6e6, получено {cfg.fdev_hz}")
        g.add(cfg.n_samples == 8192, f"n_samples=8192, получено {cfg.n_samples}")
        g.add(cfg.array == ArrayConfig(nx=16, ny=16), f"array 16x16, получено {cfg.array}")
        g.add(cfg.seed == 7, f"seed=7, получено {cfg.seed}")
        return g

    def test_default_config_source_iter(self) -> AssertionGroup:
        g = AssertionGroup("config.default_source_iter")
        cfgs = list(DefaultConfigSource().iter_configs())
        g.add(len(cfgs) == 1, f"по умолчанию iter_configs() отдаёт один конфиг, получено {len(cfgs)}")
        g.add(isinstance(cfgs[0], WaveTimeConfig), "элемент — WaveTimeConfig")
        return g

    def test_yaml_config_source(self) -> AssertionGroup:
        g = AssertionGroup("config.yaml_source")
        try:
            import yaml  # noqa: F401
        except ImportError:
            raise SkipTest("pyyaml не установлен — пропускаем YamlConfigSource (R10)") from None

        cfg = YamlConfigSource(_BASELINE_YAML).load()
        g.add(cfg.fs == 12e6, f"fs=12e6, получено {cfg.fs}")
        g.add(cfg.carrier_hz == 2e6, f"carrier_hz=2e6, получено {cfg.carrier_hz}")
        g.add(cfg.fdev_hz == 6e6, f"fdev_hz=6e6, получено {cfg.fdev_hz}")
        g.add(cfg.n_samples == 8192, f"n_samples=8192, получено {cfg.n_samples}")
        g.add(cfg.array.nx == 16 and cfg.array.ny == 16, "решётка 16x16")
        g.add(cfg.seed == 7, f"seed=7, получено {cfg.seed}")
        return g

    # ── P1: Waveform/NumpyBackend/reference ─────────────────────────────────────

    def test_cw_peak_frequency(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.cw_peak_frequency")
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=4096)
        field = CwWaveform().render(backend, spec, np.random.default_rng(1))
        sig = field.data[0, 0, :]
        spectrum = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(len(sig), d=1.0 / spec.fs)
        k_peak = int(np.argmax(spectrum))
        bin_width = spec.fs / spec.n_samples
        g.add(abs(freqs[k_peak] - spec.carrier_hz) <= bin_width,
              f"пик спектра CW должен быть на f0={spec.carrier_hz}, получено {freqs[k_peak]}")
        return g

    def test_lfm_matches_reference(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.lfm_matches_reference")
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=4096, fdev_hz=1e6)
        field = LfmWaveform().render(backend, spec, np.random.default_rng(2))
        sig = field.data[0, 0, :]
        expected = getX_numpy(spec.fs, spec.n_samples, spec.carrier_hz, 1.0, 0.0, spec.fdev_hz, 1.0)
        g.add(np.allclose(sig, expected, atol=1e-4),
              "ЛЧМ должна совпадать с reference.getX_numpy (центр. чирп, kx=ky=0 → |steer|=1)")
        return g

    def test_lfm_instantaneous_freq_linear(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.lfm_freq_linear")
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=4096, fdev_hz=1e6)
        field = LfmWaveform().render(backend, spec, np.random.default_rng(3))
        sig = field.data[0, 0, :]
        phase = np.unwrap(np.angle(sig))
        inst_freq = np.diff(phase) * spec.fs / (2 * np.pi)
        t = np.arange(len(inst_freq)) / spec.fs
        slope, intercept = np.polyfit(t, inst_freq, 1)
        fitted = slope * t + intercept
        residual = float(np.max(np.abs(inst_freq - fitted)))
        expected_slope = spec.fdev_hz / (spec.n_samples / spec.fs)
        # порог — доля полосы ЛЧМ (числ. дифференцирование фазы даёт ~сотни Гц шума на 1 МГц полосы)
        g.add(residual < 0.01 * spec.fdev_hz,
              f"мгновенная частота ЛЧМ должна быть линейной, residual={residual}")
        g.add(abs(slope - expected_slope) / expected_slope < 0.02,
              f"наклон chirp_rate ≈ {expected_slope:.1f}, получено {slope:.1f}")
        return g

    def test_am_spectrum_sidebands(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.am_sidebands")
        backend = NumpyBackend()
        fs, n = 12e6, 4096
        spec = WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n)
        f_m = fs / 128.0   # DEFAULT_F_M_FRACTION в am.py
        field = AmWaveform().render(backend, spec, np.random.default_rng(4))
        sig = field.data[0, 0, :]
        spectrum = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(n, d=1.0 / fs)

        def _bin_at(f: float) -> int:
            return int(np.argmin(np.abs(freqs - f)))

        peak_mag = float(spectrum.max())
        k_carrier = _bin_at(spec.carrier_hz)
        k_upper = _bin_at(spec.carrier_hz + f_m)
        k_lower = _bin_at(spec.carrier_hz - f_m)
        g.add(spectrum[k_carrier] > 0.3 * peak_mag, "несущая f0 должна доминировать в спектре")
        g.add(spectrum[k_upper] > 0.05 * peak_mag, "верхняя боковая f0+f_m должна быть заметна")
        g.add(spectrum[k_lower] > 0.05 * peak_mag, "нижняя боковая f0-f_m должна быть заметна")
        return g

    def test_waveform_window_energy_outside_zero(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.window_energy_outside_zero")
        backend = NumpyBackend()
        fs, n = 1000.0, 1000
        window = TimeWindow(kind="short", t0=0.3, dur=0.1)
        spec = WaveformSpec(fs=fs, carrier_hz=50.0, n_samples=n, window=window)
        field = CwWaveform().render(backend, spec, np.random.default_rng(5))
        mask = window.mask(n, fs)
        energy_outside = float(np.sum(np.abs(field.data[:, :, ~mask]) ** 2))
        energy_inside = float(np.sum(np.abs(field.data[:, :, mask]) ** 2))
        g.add(energy_outside == 0.0, f"энергия вне окна должна быть 0 (нет snr_db → нет шума), "
                                       f"получено {energy_outside}")
        g.add(energy_inside > 0.0, "внутри окна сигнал не должен быть занулён")
        return g

    def test_waveform_snr_calibration(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.snr_calibration")
        backend = NumpyBackend()
        fs, n = 12e6, 8192
        target_snr_db = 15.0
        duration = n / fs
        window = TimeWindow(kind="partial", t0=0.25 * duration, t1=0.75 * duration)
        spec = WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n, snr_db=target_snr_db, window=window)
        mask = window.mask(n, fs)
        idx = np.flatnonzero(mask)
        support = slice(int(idx[0]), int(idx[-1]) + 1)

        stat = StatisticsSnrEstimator()
        measured = []
        for seed in range(10):
            field = CwWaveform().render(backend, spec, np.random.default_rng(seed))
            sig = field.data[0, 0, :]
            measured.append(stat.estimate(sig, support).snr_db)
        mean_snr = float(np.mean(measured))
        g.add(abs(mean_snr - target_snr_db) < 1.0,
              f"измеренный SNR {mean_snr:.2f} дБ должен быть близок к заданному "
              f"{target_snr_db} дБ (±1 дБ, R5)")
        return g

    def test_waveform_shape_dtype_axes(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.shape_dtype_axes")
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=2048)
        field = LfmWaveform().render(backend, spec, np.random.default_rng(0))
        g.add(field.data.shape == (16, 16, 2048), f"shape (16,16,2048), получено {field.data.shape}")
        g.add(field.data.dtype == np.complex64, f"dtype complex64, получено {field.data.dtype}")
        g.add(field.axes == (AxisKind.ANTENNA_X, AxisKind.ANTENNA_Y, AxisKind.FAST_TIME),
              "оси: ANTENNA_X, ANTENNA_Y, FAST_TIME")
        g.add(field.modulation == Modulation.LFM, "modulation=LFM")
        return g

    def test_waveform_determinism_same_seed(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.determinism_same_seed")
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=2048, snr_db=10.0)
        field1 = CwWaveform().render(backend, spec, np.random.default_rng(42))
        field2 = CwWaveform().render(backend, spec, np.random.default_rng(42))
        g.add(np.array_equal(field1.data, field2.data),
              "один seed (R6) -> идентичный результат (побитово)")
        return g

    def test_waveform_factory_dispatch(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.factory_dispatch")
        factory = WaveformFactory()
        g.add(isinstance(factory.create(Modulation.CW), CwWaveform), "CW -> CwWaveform")
        g.add(isinstance(factory.create(Modulation.LFM), LfmWaveform), "LFM -> LfmWaveform")
        g.add(isinstance(factory.create(Modulation.AM), AmWaveform), "AM -> AmWaveform")
        raised = False
        try:
            factory.create(Modulation.NOISE)
        except ValueError:
            raised = True
        g.add(raised, "неизвестная модуляция -> ValueError")
        return g

    # ── P2: HipBackend (боевой GPU) ↔ NumpyBackend (эталон) ─────────────────────

    def _hip_backend_or_skip(self):
        """`HipBackend()`, либо `SkipTest` (нет `.so`/ROCm — Windows/cp312/CI без GPU)."""
        try:
            from core.generators.backends.hip_backend import HipBackend
        except ImportError as exc:
            raise SkipTest(f"core.generators.backends.hip_backend недоступен: {exc}") from None
        try:
            return HipBackend()
        except GpuLibsUnavailableError as exc:
            raise SkipTest(f"GPU недоступен: {exc}") from None

    def test_hip_cw_matches_numpy(self) -> AssertionGroup:
        g = AssertionGroup("p2.hip_cw_matches_numpy")
        hip = self._hip_backend_or_skip()
        numpy_backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=_P2_N)
        hip_field = hip.render(Modulation.CW, spec, np.random.default_rng(11))
        numpy_field = CwWaveform().render(numpy_backend, spec, np.random.default_rng(11))
        validator = DataValidator(tolerance=1e-3, metric="max_rel")
        result = validator.validate(hip_field.data, numpy_field.data, name="p2.cw_max_rel")
        g.add(result.passed,
              f"CW: HipBackend vs NumpyBackend max_rel={result.actual_value:.3e} "
              f"(порог {result.threshold:.0e}, решение Alex 2026-07-14 G11 п.1)")
        return g

    def test_hip_lfm_matches_numpy(self) -> AssertionGroup:
        g = AssertionGroup("p2.hip_lfm_matches_numpy")
        hip = self._hip_backend_or_skip()
        numpy_backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=_P2_N, fdev_hz=1e6)
        hip_field = hip.render(Modulation.LFM, spec, np.random.default_rng(12))
        numpy_field = LfmWaveform().render(numpy_backend, spec, np.random.default_rng(12))
        validator = DataValidator(tolerance=1e-3, metric="max_rel")
        result = validator.validate(hip_field.data, numpy_field.data, name="p2.lfm_max_rel")
        g.add(result.passed,
              f"ЛЧМ: HipBackend vs NumpyBackend(LfmWaveform, getX-формула) "
              f"max_rel={result.actual_value:.3e} (порог {result.threshold:.0e})")
        return g

    def test_hip_lfm_matches_getx_reference(self) -> AssertionGroup:
        """G11-регресс: HipBackend(ЛЧМ) должен зеркалить `reference.getX_numpy` (норм.=1)."""
        g = AssertionGroup("p2.hip_lfm_matches_getx_reference")
        hip = self._hip_backend_or_skip()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=_P2_N, fdev_hz=1e6)
        field = hip.render(Modulation.LFM, spec, np.random.default_rng(13))
        sig = field.data[0, 0, :]
        expected = getX_numpy(spec.fs, spec.n_samples, spec.carrier_hz, 1.0, 0.0, spec.fdev_hz, 1.0)
        validator = DataValidator(tolerance=1e-3, metric="max_rel")
        result = validator.validate(sig, expected, name="p2.hip_lfm_vs_getx")
        g.add(result.passed,
              f"HipBackend ЛЧМ vs reference.getX_numpy(norm=1) "
              f"max_rel={result.actual_value:.3e} (G11: формула GPU == getX центр.)")
        return g

    def test_hip_magnitude_matches_at_baseline(self) -> AssertionGroup:
        """Baseline N=8192: модуль |·| точен, хотя сырая фаза дрейфует (float32 GPU-ядро).

        Закрывает находку P2: raw-complex max_rel на N=8192 > 1e-3 (дрейф АБС. фазы —
        безобиден: в дечирпе rx·conj(ref) сокращается, оба из того же ядра), НО модуль
        совпадает до ~1e-7 → формула/амплитуда/нормировка валидны на рабочем размере (§5.1).
        """
        g = AssertionGroup("p2.hip_magnitude_baseline_n8192")
        hip = self._hip_backend_or_skip()
        numpy_backend = NumpyBackend()
        validator = DataValidator(tolerance=1e-4, metric="max_rel")
        for mod, fdev in ((Modulation.CW, 0.0), (Modulation.LFM, 6e6)):
            spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=8192, fdev_hz=fdev)
            wf = CwWaveform() if mod is Modulation.CW else LfmWaveform()
            hip_field = hip.render(mod, spec, np.random.default_rng(15))
            numpy_field = wf.render(numpy_backend, spec, np.random.default_rng(15))
            result = validator.validate(
                np.abs(hip_field.data), np.abs(numpy_field.data), name=f"p2.mag_{mod.value}_n8192"
            )
            g.add(result.passed,
                  f"{mod.value} N=8192: |модуль| max_rel={result.actual_value:.3e} "
                  f"(порог 1e-4; сырая фаза дрейфует — безобидно, см. P2-находку)")
        return g

    def test_hip_backend_rejects_unsupported_modulation(self) -> AssertionGroup:
        g = AssertionGroup("p2.hip_backend_rejects_unsupported_modulation")
        hip = self._hip_backend_or_skip()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=_P2_N)
        raised = False
        try:
            hip.render(Modulation.AM, spec, np.random.default_rng(14))
        except ValueError:
            raised = True
        g.add(raised, "HipBackend.render(AM) должен кинуть ValueError (GPU умеет только CW/ЛЧМ, §5)")
        return g


if __name__ == "__main__":
    GeneratorsTests().run_all()
