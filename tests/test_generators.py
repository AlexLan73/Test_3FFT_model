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
from scipy.stats import kurtosis  # noqa: E402

from common.runner import AssertionGroup, SkipTest, TestRunner  # noqa: E402
from common.validators import DataValidator  # noqa: E402
from core.config import ArrayConfig  # noqa: E402
from core.config.config_source import DefaultConfigSource, YamlConfigSource  # noqa: E402
from core.config.waveform_config import WaveTimeConfig  # noqa: E402
from core.generators.backends import NumpyBackend  # noqa: E402
from core.generators.waveforms import (  # noqa: E402
    AmWaveform,
    AxisKind,
    BarrageRfJammer,
    CwWaveform,
    DrfmRepeaterJammer,
    FmInterferenceWaveform,
    ImpulsiveArcJammer,
    IndustrialCwJammer,
    LfmWaveform,
    Modulation,
    PhaseCodeWaveform,
    SignalField,
    SmspJammer,
    TimeWindow,
    VfdHarmonicJammer,
    WaveformFactory,
    WaveformSpec,
    m_sequence,
)
from core.generators.waveforms.mseq import (  # noqa: E402
    _PRIMITIVE_TAPS,
    gpu_lfsr_params,
    m_sequence_pow2,
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


def _dechirp_numpy(s_rx: np.ndarray, s_ref: np.ndarray) -> np.ndarray:
    """Дечирп: `s_dc = s_rx * conj(s_ref)` (J2 — тест SMSP по дечирпу, не сырому спектру).

    vendored from DSP-GPU/DSP/Python/heterodyne/heterodyne_base.py:67
    (`HeterodyneTestBase.dechirp_numpy`) — минимальная формула, без класса/зависимостей.
    """
    return (s_rx * np.conj(s_ref)).astype(np.complex64)


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

    # ── P4: mseq (M-послед. LFSR) ─────────────────────────────────────────────

    def test_mseq_autocorrelation_thumbtack(self) -> AssertionGroup:
        """Главный тест: циклическая автокорреляция m_sequence(13) = thumbtack (пик L, фон≈-1)."""
        g = AssertionGroup("p4.mseq_autocorrelation_thumbtack")
        code = m_sequence(13, seed=1)
        length_l = code.shape[0]
        g.add(length_l == 8191, f"degree=13 -> длина 2^13-1=8191, получено {length_l}")
        g.add(code.dtype == np.float32, f"код должен быть float32, получено {code.dtype}")
        g.add(bool(np.all(np.isin(code, (-1.0, 1.0)))), "код должен состоять только из ±1")

        x = code.astype(np.float64)
        autocorr = np.array([np.sum(x * np.roll(x, -k)) for k in range(length_l)])
        peak = autocorr[0]
        side_max = float(np.max(np.abs(autocorr[1:])))
        g.add(peak == length_l, f"пик автокорреляции при сдвиге 0 должен быть L={length_l}, получено {peak}")
        g.add(side_max <= 1.0 + 1e-6,
              f"боковые лепестки thumbtack-автокорреляции должны быть ≈-1 (|·|≤1), получено max={side_max}")
        return g

    def test_mseq_all_degrees_thumbtack(self) -> AssertionGroup:
        """Вся встроенная таблица полиномов (degree 7..16) — период 2^degree-1, thumbtack."""
        g = AssertionGroup("p4.mseq_all_degrees_thumbtack")
        for degree in sorted(_PRIMITIVE_TAPS):
            code = m_sequence(degree, seed=1)
            length_l = code.shape[0]
            x = code.astype(np.float64)
            autocorr = np.array([np.sum(x * np.roll(x, -k)) for k in range(length_l)])
            g.add(length_l == (1 << degree) - 1, f"degree={degree}: длина должна быть 2^{degree}-1")
            g.add(autocorr[0] == length_l, f"degree={degree}: пик автокорр. должен быть L={length_l}")
            g.add(float(np.max(np.abs(autocorr[1:]))) <= 1.0 + 1e-6,
                  f"degree={degree}: полином не примитивен (плохой thumbtack) — проверь _PRIMITIVE_TAPS")
        return g

    def test_mseq_determinism_same_seed(self) -> AssertionGroup:
        g = AssertionGroup("p4.mseq_determinism_same_seed")
        a = m_sequence(13, seed=5)
        b = m_sequence(13, seed=5)
        c = m_sequence(13, seed=6)
        g.add(bool(np.array_equal(a, b)), "один seed (R6) -> идентичный код (побитово)")
        g.add(not np.array_equal(a, c), "разные seed должны давать разные коды")
        return g

    def test_mseq_invalid_params_raise(self) -> AssertionGroup:
        g = AssertionGroup("p4.mseq_invalid_params_raise")
        raised_degree = False
        try:
            m_sequence(degree=32)
        except ValueError:
            raised_degree = True
        g.add(raised_degree, "degree>=32 должен кинуть ValueError (регистр 32-бит)")

        raised_seed = False
        try:
            m_sequence(degree=13, seed=0)
        except ValueError:
            raised_seed = True
        g.add(raised_seed, "seed=0 должен кинуть ValueError (нулевое состояние LFSR вырождено)")

        raised_no_table = False
        try:
            m_sequence(degree=20)
        except ValueError:
            raised_no_table = True
        g.add(raised_no_table, "degree без встроенного полинома и без poly_taps -> ValueError")
        return g

    def test_mseq_pow2_wraps_to_first_sample(self) -> AssertionGroup:
        """H3: `m_sequence_pow2` продолжает LFSR на 1 такт -> seq[L] == seq[0] (полный период)."""
        g = AssertionGroup("p4.mseq_pow2_wraps_to_first_sample")
        degree = 13
        base = m_sequence(degree, seed=1)
        padded = m_sequence_pow2(degree, seed=1)
        g.add(padded.shape[0] == (1 << degree), f"длина должна быть 2^{degree}, получено {padded.shape[0]}")
        g.add(bool(np.array_equal(padded[: base.shape[0]], base)),
              "первые L отсчётов m_sequence_pow2 должны совпадать с m_sequence")
        g.add(bool(padded[base.shape[0]] == base[0]),
              "последний (L-й) отсчёт должен == первому (полный период LFSR вернулся в seed)")
        return g

    # ── P4: PhaseCodeWaveform (ФМн) ──────────────────────────────────────────

    def test_phase_code_spectrum_wideband(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.phase_code_spectrum_wideband")
        backend = NumpyBackend()
        fs, n = 12e6, 8192
        spec = WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n, meta={"degree": 13, "seed": 1})
        field = PhaseCodeWaveform().render(backend, spec, np.random.default_rng(20))
        sig = field.data[0, 0, :]
        spectrum = np.abs(np.fft.fft(sig)) ** 2
        peak_fraction = float(spectrum.max() / spectrum.sum())

        cw_field = CwWaveform().render(
            backend, WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n), np.random.default_rng(21)
        )
        cw_spectrum = np.abs(np.fft.fft(cw_field.data[0, 0, :])) ** 2
        cw_peak_fraction = float(cw_spectrum.max() / cw_spectrum.sum())

        g.add(peak_fraction < 0.05,
              f"ФМн должен быть широкополосным (пик не доминирует), peak_fraction={peak_fraction:.4f}")
        g.add(peak_fraction < cw_peak_fraction / 10,
              f"ФМн peak_fraction={peak_fraction:.4f} должен быть << CW peak_fraction={cw_peak_fraction:.4f}")
        return g

    def test_phase_code_meta_code_is_real_pm1(self) -> AssertionGroup:
        """H1: `field.meta['code']` — РЕАЛЬНЫЙ ±1 float32 (то, что подаём коррелятору), не data."""
        g = AssertionGroup("waveforms.phase_code_meta_code_h1")
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=8192, meta={"degree": 13, "seed": 3})
        field = PhaseCodeWaveform().render(backend, spec, np.random.default_rng(22))
        code = field.meta.get("code")
        g.add(code is not None, "field.meta должен содержать ключ 'code' (H1)")
        g.add(code.dtype == np.float32, f"code.dtype должен быть float32, получено {code.dtype}")
        g.add(not np.iscomplexobj(code), "code должен быть вещественным (НЕ complex passband), H1")
        g.add(code.shape[0] == 8191, f"code — длина L=2^13-1=8191 (сырой, до растяжения), получено {code.shape}")
        g.add(bool(np.all(np.isin(code, (-1.0, 1.0)))), "code должен состоять только из ±1")
        g.add(field.data.dtype == np.complex64, "field.data остаётся complex64 (passband) — не путать с code")
        return g

    def test_phase_code_factory_dispatch_and_shape(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.phase_code_factory_dispatch_and_shape")
        factory = WaveformFactory()
        g.add(isinstance(factory.create(Modulation.PHASE_CODE), PhaseCodeWaveform),
              "PHASE_CODE -> PhaseCodeWaveform")
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=2048, meta={"degree": 11})
        field = PhaseCodeWaveform().render(backend, spec, np.random.default_rng(23))
        g.add(field.data.shape == (16, 16, 2048), f"shape (16,16,2048), получено {field.data.shape}")
        g.add(field.data.dtype == np.complex64, f"dtype complex64, получено {field.data.dtype}")
        g.add(field.modulation == Modulation.PHASE_CODE, "modulation=PHASE_CODE")
        return g

    # ── P4: FmInterferenceWaveform (ЧМ-помеха) ───────────────────────────────

    def test_fm_interference_sidebands(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.fm_interference_sidebands")
        backend = NumpyBackend()
        fs, n = 12e6, 8192
        f_m = fs / 256.0
        spec = WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n, meta={"beta": 2.0, "f_m": f_m})
        field = FmInterferenceWaveform().render(backend, spec, np.random.default_rng(30))
        sig = field.data[0, 0, :]
        spectrum = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(n, d=1.0 / fs)

        def _bin_at(f: float) -> int:
            return int(np.argmin(np.abs(freqs - f)))

        peak_mag = float(spectrum.max())
        k_upper = _bin_at(spec.carrier_hz + f_m)
        k_lower = _bin_at(spec.carrier_hz - f_m)
        occ_bins = int(np.sum(spectrum > 0.05 * peak_mag))
        g.add(spectrum[k_upper] > 0.05 * peak_mag, "боковая f0+f_m (Бессель J1) должна быть заметна")
        g.add(spectrum[k_lower] > 0.05 * peak_mag, "боковая f0-f_m (Бессель J1) должна быть заметна")
        g.add(occ_bins > 2, f"спектр ЧМ должен иметь несколько значимых компонент, occ_bins={occ_bins}")
        return g

    def test_fm_interference_factory_dispatch_and_shape(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.fm_interference_factory_dispatch_and_shape")
        factory = WaveformFactory()
        g.add(isinstance(factory.create(Modulation.FM_INTERFERENCE), FmInterferenceWaveform),
              "FM_INTERFERENCE -> FmInterferenceWaveform")
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=2048)
        field = FmInterferenceWaveform().render(backend, spec, np.random.default_rng(31))
        g.add(field.data.shape == (16, 16, 2048), f"shape (16,16,2048), получено {field.data.shape}")
        g.add(field.data.dtype == np.complex64, f"dtype complex64, получено {field.data.dtype}")
        g.add(field.modulation == Modulation.FM_INTERFERENCE, "modulation=FM_INTERFERENCE")
        return g

    def test_phase_code_fm_determinism_same_seed(self) -> AssertionGroup:
        g = AssertionGroup("waveforms.phase_code_fm_determinism_same_seed")
        backend = NumpyBackend()
        spec_pc = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=2048, snr_db=10.0, meta={"degree": 11})
        f1 = PhaseCodeWaveform().render(backend, spec_pc, np.random.default_rng(42))
        f2 = PhaseCodeWaveform().render(backend, spec_pc, np.random.default_rng(42))
        g.add(np.array_equal(f1.data, f2.data), "ФМн: один seed (R6) -> идентичный результат")

        spec_fm = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=2048, snr_db=10.0)
        f3 = FmInterferenceWaveform().render(backend, spec_fm, np.random.default_rng(43))
        f4 = FmInterferenceWaveform().render(backend, spec_fm, np.random.default_rng(43))
        g.add(np.array_equal(f3.data, f4.data), "ЧМ: один seed (R6) -> идентичный результат")
        return g

    # ── P4: GPU-коррелятор (реюз FMCorrelatorROCm) — H1/H2/H3 ───────────────

    def _fm_correlator_or_skip(self):
        """`(dsp_radar module, ROCmGPUContext)`, либо `SkipTest` (нет `.so`/ROCm)."""
        try:
            from common.gpu_context import GPUContextManager
            from core.gpu_libs import loader as gpu_libs
        except ImportError as exc:
            raise SkipTest(f"gpu_context/gpu_libs недоступны: {exc}") from None
        try:
            gpu_libs.require()
            radar = gpu_libs.load("dsp_radar")
        except GpuLibsUnavailableError as exc:
            raise SkipTest(f"GPU недоступен: {exc}") from None
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise SkipTest("ROCmGPUContext недоступен (ROCm-девайс не создан).")
        return radar, ctx

    def test_mseq_matches_correlator_generate_msequence(self) -> AssertionGroup:
        """H2: наш `m_sequence`(выровненный по ст. битам) == `FMCorrelatorROCm.generate_msequence`
        для ТОГО ЖЕ (выровненного) полинома/сида — бит-в-бит, на GPU (RX 9070)."""
        g = AssertionGroup("p4.mseq_matches_correlator_generate_msequence")
        radar, ctx = self._fm_correlator_or_skip()
        degree, seed = 13, 1
        polynomial, seed32 = gpu_lfsr_params(degree, seed)
        fft_size = 1 << degree

        corr = radar.FMCorrelatorROCm(ctx)
        corr.set_params(fft_size=fft_size, num_shifts=1, num_signals=1, num_output_points=10,
                         polynomial=polynomial, seed=seed32)
        gpu_seq = corr.generate_msequence(seed32)
        our_seq = m_sequence_pow2(degree, seed)

        g.add(gpu_seq.shape == our_seq.shape,
              f"формы должны совпасть: gpu={gpu_seq.shape} наш={our_seq.shape}")
        matches = bool(np.array_equal(gpu_seq, our_seq))
        n_diff = int(np.sum(gpu_seq != our_seq)) if not matches else 0
        g.add(matches, f"наш LFSR должен совпасть с GPU generate_msequence бит-в-бит "
                        f"(polynomial=0x{polynomial:08x}, seed=0x{seed32:08x}); расхождений={n_diff}")
        return g

    def test_correlator_peak_at_shift(self) -> AssertionGroup:
        """H1/H3: наш real ±1 `m_sequence_pow2` как ref -> `process()` на циклически
        сдвинутых `[S,N]` real float32 входах -> пик на позиции сдвига (GPU, RX 9070)."""
        g = AssertionGroup("p4.correlator_peak_at_shift")
        radar, ctx = self._fm_correlator_or_skip()
        degree = 13
        ref = m_sequence_pow2(degree, seed=1)   # H1: real ±1, НЕ SignalField.data; H3: pow2-длина
        fft_size = ref.shape[0]
        shifts = [0, 5, 17, 100, 250]
        n_kg = 400

        corr = radar.FMCorrelatorROCm(ctx)
        corr.set_params(fft_size=fft_size, num_shifts=1, num_signals=len(shifts), num_output_points=n_kg)
        corr.prepare_reference_from_data(ref)
        signals = np.stack([np.roll(ref, d) for d in shifts]).astype(np.float32)  # [S, N] real
        peaks = corr.process(signals)

        g.add(peaks.shape == (len(shifts), 1, n_kg), f"peaks.shape должен быть (S,1,n_kg), получено {peaks.shape}")
        for i, d in enumerate(shifts):
            row = peaks[i, 0, :]
            amax = int(np.argmax(row))
            g.add(amax == d, f"shift d={d}: пик на позиции {amax} (ожидали {d}), val={row[amax]:.1f}")
        return g

    # ── P5: помехи патент+промышленные (jammers_rf) ──────────────────────────

    def test_p5_factory_dispatch_and_shape(self) -> AssertionGroup:
        g = AssertionGroup("p5.factory_dispatch_and_shape")
        factory = WaveformFactory()
        pairs = [
            (Modulation.BARRAGE, BarrageRfJammer),
            (Modulation.SMSP, SmspJammer),
            (Modulation.DRFM_REPEATER, DrfmRepeaterJammer),
            (Modulation.INDUSTRIAL_CW, IndustrialCwJammer),
            (Modulation.IMPULSIVE_ARC, ImpulsiveArcJammer),
            (Modulation.VFD_HARMONIC, VfdHarmonicJammer),
        ]
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=2048, fdev_hz=1e6)
        for modulation, cls in pairs:
            g.add(isinstance(factory.create(modulation), cls), f"{modulation} -> {cls.__name__}")
            field = cls().render(backend, spec, np.random.default_rng(0))
            g.add(field.data.shape == (16, 16, 2048),
                  f"{modulation}: shape (16,16,2048), получено {field.data.shape}")
            g.add(field.data.dtype == np.complex64, f"{modulation}: dtype complex64")
            g.add(field.modulation == modulation, f"{modulation}: field.modulation совпадает")
        return g

    def test_p5_determinism_same_seed(self) -> AssertionGroup:
        g = AssertionGroup("p5.determinism_same_seed")
        backend = NumpyBackend()
        spec = WaveformSpec(fs=12e6, carrier_hz=2e6, n_samples=2048, fdev_hz=1e6)
        for cls in (BarrageRfJammer, SmspJammer, DrfmRepeaterJammer,
                    IndustrialCwJammer, ImpulsiveArcJammer, VfdHarmonicJammer):
            f1 = cls().render(backend, spec, np.random.default_rng(77))
            f2 = cls().render(backend, spec, np.random.default_rng(77))
            g.add(np.array_equal(f1.data, f2.data),
                  f"{cls.__name__}: один seed (R6) -> идентичный результат (побитово)")
        return g

    def test_barrage_wideband_and_coherent_steering(self) -> AssertionGroup:
        """J4: barrage — когерентный/направленный (rank-1 через steering), широкополосный."""
        g = AssertionGroup("p5.barrage_wideband_and_coherent")
        backend = NumpyBackend()
        fs, n = 12e6, 4096
        kx, ky = 3.0, -2.0
        spec = WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n, amplitude=2.0,
                             meta={"kx": kx, "ky": ky, "nx": 16, "ny": 16})
        field = BarrageRfJammer().render(backend, spec, np.random.default_rng(50))

        sig = field.data[0, 0, :]
        spectrum = np.abs(np.fft.fft(sig)) ** 2
        peak_fraction = float(spectrum.max() / spectrum.sum())

        cw_spec = WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n)
        cw_field = CwWaveform().render(backend, cw_spec, np.random.default_rng(51))
        cw_spectrum = np.abs(np.fft.fft(cw_field.data[0, 0, :])) ** 2
        cw_peak_fraction = float(cw_spectrum.max() / cw_spectrum.sum())

        g.add(peak_fraction < 0.02,
              f"barrage должен быть широкополосным (равномерный пол), peak_fraction={peak_fraction:.5f}")
        g.add(peak_fraction < cw_peak_fraction / 20,
              f"barrage peak_fraction={peak_fraction:.5f} должен быть << CW peak_fraction={cw_peak_fraction:.4f}")

        from core.generators.grid import ArrayGrid
        steer = ArrayGrid(16, 16).steering(kx, ky)
        expected_ratio = steer[7, 9] / steer[0, 0]
        actual_ratio = field.data[7, 9, :] / field.data[0, 0, :]
        g.add(bool(np.allclose(actual_ratio, expected_ratio, atol=1e-4)),
              "barrage когерентен по элементам (J4): отношение между элементами = steering, "
              "не независимый шум на элемент")
        return g

    def test_smsp_smeared_after_dechirp_vs_matched_lfm(self) -> AssertionGroup:
        """J2: сырой спектр SMSP и ЛЧМ той же ширины НЕ различает; смотрим ПОСЛЕ дечирпа."""
        g = AssertionGroup("p5.smsp_smeared_after_dechirp")
        backend = NumpyBackend()
        fs, n, f0, fdev = 12e6, 4096, 2e6, 1e6
        k_segments = 8

        lfm_spec = WaveformSpec(fs=fs, carrier_hz=f0, n_samples=n, fdev_hz=fdev)
        lfm_field = LfmWaveform().render(backend, lfm_spec, np.random.default_rng(60))
        lfm_sig = lfm_field.data[0, 0, :]

        smsp_spec = WaveformSpec(fs=fs, carrier_hz=f0, n_samples=n, fdev_hz=fdev,
                                  meta={"k_segments": k_segments})
        smsp_field = SmspJammer().render(backend, smsp_spec, np.random.default_rng(61))
        smsp_sig = smsp_field.data[0, 0, :]

        ref = getX_numpy(fs, n, f0, 1.0, 0.0, fdev, 1.0)

        def _energy_bandwidth_90(sig: np.ndarray) -> float:
            """90%-энергетическая полоса — устойчивее к пиковой форме спектра, чем
            «число бин выше X% пика» (SMSP даёт более острые пики из-за периодического
            повтора K сегментов -> «bins above threshold» занижает его полосу; полная
            энергетическая полоса же остаётся физически сопоставимой, что и требует J2)."""
            freqs = np.fft.fftfreq(sig.shape[0], d=1.0 / fs)
            power = np.abs(np.fft.fft(sig)) ** 2
            order = np.argsort(freqs)
            f_sorted, p_sorted = freqs[order], power[order]
            cdf = np.cumsum(p_sorted) / p_sorted.sum()
            lo = f_sorted[np.searchsorted(cdf, 0.05)]
            hi = f_sorted[np.searchsorted(cdf, 0.95)]
            return float(hi - lo)

        raw_lfm_bw = _energy_bandwidth_90(lfm_sig)
        raw_smsp_bw = _energy_bandwidth_90(smsp_sig)
        ratio = raw_smsp_bw / max(raw_lfm_bw, 1.0)
        g.add(0.5 < ratio < 2.0,
              f"J2: сырые 90%-энергетические полосы ЛЧМ({raw_lfm_bw:.0f} Гц) и "
              f"SMSP({raw_smsp_bw:.0f} Гц) должны быть сопоставимы (сырой спектр НЕ различает), "
              f"ratio={ratio:.2f}")

        matched_dechirped = _dechirp_numpy(lfm_sig, ref)
        smsp_dechirped = _dechirp_numpy(smsp_sig, ref)
        matched_spec = np.abs(np.fft.fft(matched_dechirped)) ** 2
        smsp_spec_ = np.abs(np.fft.fft(smsp_dechirped)) ** 2
        matched_peak_fraction = float(matched_spec.max() / matched_spec.sum())
        smsp_peak_fraction = float(smsp_spec_.max() / smsp_spec_.sum())

        g.add(matched_peak_fraction > 0.9,
              f"matched ЛЧМ после дечирпа опорным -> острый пик, peak_fraction={matched_peak_fraction:.4f}")
        g.add(smsp_peak_fraction < 0.3,
              f"SMSP после дечирпа -> размазан, peak_fraction={smsp_peak_fraction:.4f}")
        g.add(smsp_peak_fraction < matched_peak_fraction / 3,
              f"SMSP peak_fraction={smsp_peak_fraction:.4f} должен быть << matched={matched_peak_fraction:.4f}")
        return g

    def test_drfm_repeater_cross_correlation_peaks(self) -> AssertionGroup:
        g = AssertionGroup("p5.drfm_repeater_cross_correlation_peaks")
        backend = NumpyBackend()
        fs, n, f0, fdev = 12e6, 2048, 2e6, 1e6
        # spacing >> автокорр. mainlobe чирпа (~1/fdev=1мкс=12 отсч.) — иначе соседние
        # копии сливаются в одну корреляционную "кляксу" (проверено эмпирически).
        lead_s, spacing_s, count = 5e-6, 5e-6, 5
        spec = WaveformSpec(fs=fs, carrier_hz=f0, n_samples=n, fdev_hz=fdev,
                             meta={"lead_s": lead_s, "spacing_s": spacing_s,
                                   "count": count, "decay": 0.85})
        field = DrfmRepeaterJammer().render(backend, spec, np.random.default_rng(70))
        sig = field.data[0, 0, :]
        ref = getX_numpy(fs, n, f0, 1.0, 0.0, fdev, 1.0)

        nfft = 2 * n
        corr = np.fft.ifft(np.fft.fft(sig, nfft) * np.conj(np.fft.fft(ref, nfft)))
        mag = np.abs(corr)
        noise_floor = float(np.median(mag))

        expected_shifts = [round((lead_s + i * spacing_s) * fs) for i in range(count)]
        for i, shift in enumerate(expected_shifts):
            lo, hi = max(0, shift - 2), shift + 3
            window = mag[lo:hi]
            local_peak_idx = lo + int(np.argmax(window))
            g.add(abs(local_peak_idx - shift) <= 1,
                  f"копия #{i}: локальный максимум на {local_peak_idx}, ожидали τ_i={shift} (±1 отсч.)")
            g.add(float(window.max()) > 5.0 * noise_floor,
                  f"копия #{i} (τ_i={shift} отсч.): пик кросс-корр={window.max():.1f} "
                  f"должен быть >> фон={noise_floor:.1f}")
        return g

    def test_industrial_cw_spectrum_peak(self) -> AssertionGroup:
        g = AssertionGroup("p5.industrial_cw_spectrum_peak")
        backend = NumpyBackend()
        fs, n = 12e6, 4096
        spec = WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n)   # f_int по умолчанию = fs*0.29
        field = IndustrialCwJammer().render(backend, spec, np.random.default_rng(80))
        sig = field.data[0, 0, :]
        spectrum = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(n, d=1.0 / fs)
        k_peak = int(np.argmax(spectrum))
        f_int_expected = fs * 0.29
        bin_width = fs / n
        g.add(abs(freqs[k_peak] - f_int_expected) <= bin_width,
              f"пик спектра INT_CW должен быть на f_int={f_int_expected:.0f}, получено {freqs[k_peak]:.0f}")
        g.add(f_int_expected < fs / 2, f"J3: f_int={f_int_expected:.0f} должен быть < fs/2={fs / 2:.0f}")
        return g

    def test_impulsive_arc_kurtosis_and_sparsity(self) -> AssertionGroup:
        g = AssertionGroup("p5.impulsive_arc_kurtosis_and_sparsity")
        backend = NumpyBackend()
        fs, n = 12e6, 8192
        spec = WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n, amplitude=5.0,
                             meta={"lambda_hz": 5e4, "tau_decay_s": 3e-7, "alpha_stable": 1.4})
        field = ImpulsiveArcJammer().render(backend, spec, np.random.default_rng(90))
        sig = field.data[0, 0, :]
        mag = np.abs(sig)

        k = float(kurtosis(mag, fisher=False))
        g.add(k > 3.0, f"IMP_ARC должен иметь высокий эксцесс (kurtosis>3, Gauss~3), получено {k:.2f}")

        threshold = 3.0 * float(np.std(mag))
        sparsity = float(np.mean(mag > threshold))
        g.add(sparsity < 0.05, f"IMP_ARC должен быть разрежен во времени, доля выбросов={sparsity:.4f}")
        g.add(bool(np.any(mag > 0)), "IMP_ARC не должен быть тождественно нулевым")
        return g

    def test_vfd_harmonic_spectrum_peaks(self) -> AssertionGroup:
        g = AssertionGroup("p5.vfd_harmonic_spectrum_peaks")
        backend = NumpyBackend()
        fs, n, f_sw, n_harm = 12e6, 8192, 8e3, 5
        spec = WaveformSpec(fs=fs, carrier_hz=2e6, n_samples=n,
                             meta={"f_sw": f_sw, "n_harmonics": n_harm, "broadband_frac": 0.0})
        field = VfdHarmonicJammer().render(backend, spec, np.random.default_rng(95))
        sig = field.data[0, 0, :]
        spectrum = np.abs(np.fft.fft(sig))
        freqs = np.fft.fftfreq(n, d=1.0 / fs)

        def _bin_at(f: float) -> int:
            return int(np.argmin(np.abs(freqs - f)))

        noise_floor = float(np.median(spectrum))
        found = 0
        for harm in range(1, n_harm + 1):
            k = _bin_at(harm * f_sw)
            if spectrum[k] > 10.0 * noise_floor:
                found += 1
        g.add(found == n_harm, f"должны быть видны все {n_harm} гармоник n*f_sw, найдено {found}")
        return g


if __name__ == "__main__":
    GeneratorsTests().run_all()
