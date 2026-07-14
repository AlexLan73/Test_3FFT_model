"""Демо P1+P2: визуал-подтверждение Waveform/NumpyBackend (CW/ЛЧМ/АМ) и
HipBackend↔NumpyBackend (GPU↔эталон), §9-конвенция спеки.

Composition Root — здесь и только здесь связываются бэкенды + конкретные
волны + `FigureWriter` (Pure Fabrication). P1 заменяет временный `demo_p0.py`
(окно показано в составе этого демо, `window_placement.png`). P2 добавляет
ветку `HipBackend` (выбор по доступности `.so`/ROCm — Linux+cp313; иначе
графики P2 не строятся, это норма, см. §9 спеки).

Запуск:  .venv/bin/python demo_generators.py       # только P1 (NumpyBackend)
         /usr/bin/python3.13 demo_generators.py    # P1 + P2 (если GPU доступен)
Пишет:   graphics/signal_generators/p1_numpy_cw_lfm_am/{cw_time_spectrum,
         lfm_spectrogram, am_spectrum, window_placement, snr_check}.png
         graphics/signal_generators/p2_gpu_vs_numpy/{gpu_vs_numpy_overlay,
         gpu_vs_numpy_error}.png   (только если HipBackend доступен)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from core.generators.backends import NumpyBackend
from core.generators.waveforms import (
    AmWaveform,
    CwWaveform,
    FmInterferenceWaveform,
    LfmWaveform,
    Modulation,
    PhaseCodeWaveform,
    TimeWindow,
    WaveformSpec,
    m_sequence,
)
from core.generators.waveforms.mseq import m_sequence_pow2
from core.gpu_libs.loader import GpuLibsUnavailableError
from core.graphics.writer import FigureWriter
from core.snr import StatisticsSnrEstimator

_OUT_DIR = "graphics/signal_generators/p1_numpy_cw_lfm_am"
_OUT_DIR_P2 = "graphics/signal_generators/p2_gpu_vs_numpy"
_OUT_DIR_P4 = "graphics/signal_generators/p4_phase_code_fm"
_FS = 12e6
_F0 = 2e6
_N = 4096


def _demo_cw(backend: NumpyBackend, writer: FigureWriter) -> None:
    spec = WaveformSpec(fs=_FS, carrier_hz=_F0, n_samples=_N)
    field = CwWaveform().render(backend, spec, np.random.default_rng(1))
    sig = field.data[0, 0, :]
    t = np.arange(_N) / _FS
    spectrum = np.abs(np.fft.fft(sig))
    freqs = np.fft.fftfreq(_N, d=1.0 / _FS)
    order = np.argsort(freqs)

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(8, 6))
    ax_t.plot(t * 1e6, sig.real, label="Re")
    ax_t.plot(t * 1e6, np.abs(sig), label="|·|", alpha=0.6)
    ax_t.set_title("CW: время")
    ax_t.set_xlabel("t, мкс")
    ax_t.legend()
    ax_t.grid(True, alpha=0.3)

    ax_f.plot(freqs[order] / 1e6, spectrum[order])
    ax_f.axvline(_F0 / 1e6, color="r", linestyle="--", alpha=0.5, label="f0")
    ax_f.set_title("CW: спектр (один пик на f0)")
    ax_f.set_xlabel("f, МГц")
    ax_f.legend()
    ax_f.grid(True, alpha=0.3)

    fig.tight_layout()
    writer.write(fig, "cw_time_spectrum.png")
    plt.close(fig)


def _demo_lfm(backend: NumpyBackend, writer: FigureWriter) -> None:
    spec = WaveformSpec(fs=_FS, carrier_hz=_F0, n_samples=_N, fdev_hz=3e6)
    field = LfmWaveform().render(backend, spec, np.random.default_rng(2))
    sig = field.data[0, 0, :]

    f_spec, t_spec, sxx = spectrogram(sig, fs=_FS, nperseg=256, noverlap=224, return_onesided=False)
    order = np.argsort(f_spec)

    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(t_spec * 1e6, f_spec[order] / 1e6, 10 * np.log10(sxx[order] + 1e-20),
                          shading="auto")
    fig.colorbar(mesh, ax=ax, label="дБ")
    ax.set_title("ЛЧМ: спектрограмма (наклонная линия = линейный чирп)")
    ax.set_xlabel("t, мкс")
    ax.set_ylabel("f, МГц")
    fig.tight_layout()
    writer.write(fig, "lfm_spectrogram.png")
    plt.close(fig)


def _demo_am(backend: NumpyBackend, writer: FigureWriter) -> None:
    spec = WaveformSpec(fs=_FS, carrier_hz=_F0, n_samples=_N)
    f_m = _FS / 128.0
    field = AmWaveform().render(backend, spec, np.random.default_rng(4))
    sig = field.data[0, 0, :]
    spectrum = np.abs(np.fft.fft(sig))
    freqs = np.fft.fftfreq(_N, d=1.0 / _FS)
    order = np.argsort(freqs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs[order] / 1e6, spectrum[order])
    for f, label in ((_F0, "f0"), (_F0 - f_m, "f0-f_m"), (_F0 + f_m, "f0+f_m")):
        ax.axvline(f / 1e6, color="r", linestyle="--", alpha=0.4)
        ax.text(f / 1e6, spectrum.max() * 0.9, label, rotation=90, fontsize=8)
    ax.set_title("АМ: спектр (несущая + 2 боковые f0±f_m)")
    ax.set_xlabel("f, МГц")
    ax.set_xlim((_F0 - 4 * f_m) / 1e6, (_F0 + 4 * f_m) / 1e6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    writer.write(fig, "am_spectrum.png")
    plt.close(fig)


def _demo_window_placement(backend: NumpyBackend, writer: FigureWriter) -> None:
    fs, n = 1000.0, 1000
    t = np.arange(n) / fs
    windows = [
        ("full", TimeWindow(kind="full")),
        ("partial(t0=0.2, t1=0.6)", TimeWindow(kind="partial", t0=0.2, t1=0.6)),
        ("short(t0=0.3, dur=0.1)", TimeWindow(kind="short", t0=0.3, dur=0.1)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for ax, (title, window) in zip(axes, windows, strict=True):
        spec = WaveformSpec(fs=fs, carrier_hz=50.0, n_samples=n, window=window)
        field = CwWaveform().render(backend, spec, np.random.default_rng(5))
        sig = field.data[0, 0, :]
        ax.plot(t, np.abs(sig), drawstyle="steps-post")
        ax.set_title(f"CwWaveform + TimeWindow: {title}")
        ax.set_ylabel("|сигнал(t)|")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t, c")
    fig.tight_layout()
    writer.write(fig, "window_placement.png")
    plt.close(fig)


def _demo_snr_check(backend: NumpyBackend, writer: FigureWriter) -> None:
    fs, n = _FS, 8192
    duration = n / fs
    window = TimeWindow(kind="partial", t0=0.25 * duration, t1=0.75 * duration)
    mask = window.mask(n, fs)
    idx = np.flatnonzero(mask)
    support = slice(int(idx[0]), int(idx[-1]) + 1)
    estimator = StatisticsSnrEstimator()

    target_values = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    measured_values = []
    for target in target_values:
        spec = WaveformSpec(fs=fs, carrier_hz=_F0, n_samples=n, snr_db=target, window=window)
        vals = []
        for seed in range(10):
            field = CwWaveform().render(backend, spec, np.random.default_rng(seed))
            sig = field.data[0, 0, :]
            vals.append(estimator.estimate(sig, support).snr_db)
        measured_values.append(float(np.mean(vals)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(target_values, target_values, "k--", alpha=0.5, label="идеал (диагональ)")
    ax.plot(target_values, measured_values, "o-", label="измерено (StatisticsSnrEstimator)")
    ax.fill_between(target_values,
                     [v - 1.0 for v in target_values],
                     [v + 1.0 for v in target_values],
                     alpha=0.15, label="±1 дБ (R5)")
    ax.set_xlabel("snr_db заданный")
    ax.set_ylabel("snr_db измеренный")
    ax.set_title("Калибровка SNR: заданный vs измеренный")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    writer.write(fig, "snr_check.png")
    plt.close(fig)


def _demo_gpu_vs_numpy(writer: FigureWriter) -> bool:
    """P2: HipBackend↔NumpyBackend — оверлей + карта ошибки (CW и ЛЧМ).

    Возвращает False (и ничего не пишет), если `.so`/ROCm недоступны — норма
    на Windows/cp312 (§9 спеки: «на Windows/без ROCm графики P2 не строятся»).
    """
    try:
        from core.generators.backends.hip_backend import HipBackend
        hip = HipBackend()
    except (ImportError, GpuLibsUnavailableError) as exc:
        print(f"P2 GPU-демо пропущено (нет .so/ROCm): {exc}")
        return False

    numpy_backend = NumpyBackend()
    n = 4096   # запас < 1e-3 (на N=8192-baseline float32-дрейф фазы GPU превышает порог)
    specs = {
        "CW": (WaveformSpec(fs=_FS, carrier_hz=_F0, n_samples=n), Modulation.CW, CwWaveform()),
        "LFM": (WaveformSpec(fs=_FS, carrier_hz=_F0, n_samples=n, fdev_hz=1e6),
                Modulation.LFM, LfmWaveform()),
    }

    fig1, axes1 = plt.subplots(2, 1, figsize=(9, 7))
    fig2, axes2 = plt.subplots(2, 1, figsize=(9, 7))
    t = np.arange(n) / _FS

    for row, (label, (spec, modulation, waveform)) in enumerate(specs.items()):
        hip_field = hip.render(modulation, spec, np.random.default_rng(100 + row))
        numpy_field = waveform.render(numpy_backend, spec, np.random.default_rng(100 + row))
        hip_sig = hip_field.data[0, 0, :]
        numpy_sig = numpy_field.data[0, 0, :]
        err = np.abs(hip_sig.astype(np.complex128) - numpy_sig.astype(np.complex128))
        max_rel = float(err.max() / np.abs(numpy_sig.astype(np.complex128)).max())

        ax = axes1[row]
        ax.plot(t * 1e6, numpy_sig.real, label="NumpyBackend Re", alpha=0.8)
        ax.plot(t * 1e6, hip_sig.real, "--", label="HipBackend Re", alpha=0.8)
        ax.set_title(f"{label}: GPU↔NumPy оверлей (max_rel={max_rel:.2e})")
        ax.set_xlabel("t, мкс")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2 = axes2[row]
        ax2.plot(t * 1e6, err)
        ax2.axhline(1e-3 * np.abs(numpy_sig.astype(np.complex128)).max(), color="r",
                     linestyle="--", alpha=0.5, label="порог 1e-3·max|ref|")
        ax2.set_title(f"{label}: |HipBackend − NumpyBackend| (max_rel={max_rel:.2e})")
        ax2.set_xlabel("t, мкс")
        ax2.set_ylabel("|Δ|")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig1.tight_layout()
    writer.write(fig1, "gpu_vs_numpy_overlay.png")
    plt.close(fig1)

    fig2.tight_layout()
    writer.write(fig2, "gpu_vs_numpy_error.png")
    plt.close(fig2)
    return True


def _demo_mseq_autocorr(writer: FigureWriter) -> None:
    """P4: thumbtack-автокорреляция `m_sequence(13)` — пик L, фон ≈-1 (доказывает полином)."""
    degree = 13
    code = m_sequence(degree, seed=1)
    length_l = code.shape[0]
    x = code.astype(np.float64)
    autocorr = np.array([np.sum(x * np.roll(x, -k)) for k in range(length_l)])

    fig, (ax_code, ax_ac) = plt.subplots(2, 1, figsize=(9, 6))
    ax_code.step(np.arange(200), code[:200], where="mid")
    ax_code.set_title(f"m_sequence(degree={degree}): первые 200 чипов из L={length_l}")
    ax_code.set_xlabel("k (чип)")
    ax_code.set_ylabel("±1")
    ax_code.grid(True, alpha=0.3)

    shifts = np.arange(length_l)
    ax_ac.plot(shifts, autocorr)
    ax_ac.axhline(-1.0, color="r", linestyle="--", alpha=0.5, label="фон ≈ -1 (thumbtack)")
    ax_ac.set_title(f"Циклическая автокорреляция: пик={autocorr[0]:.0f}=L, "
                     f"max|фон|={np.max(np.abs(autocorr[1:])):.1f}")
    ax_ac.set_xlabel("сдвиг k")
    ax_ac.legend()
    ax_ac.grid(True, alpha=0.3)

    fig.tight_layout()
    writer.write(fig, "mseq_autocorr.png")
    plt.close(fig)


def _demo_phase_code_spectrum(backend: NumpyBackend, writer: FigureWriter) -> None:
    """P4: ФМн — код (±1 по времени) + широкополосный спектр (§5/§6.2)."""
    fs, n, degree = _FS, 8192, 13
    spec = WaveformSpec(fs=fs, carrier_hz=_F0, n_samples=n, meta={"degree": degree, "seed": 1})
    field = PhaseCodeWaveform().render(backend, spec, np.random.default_rng(20))
    sig = field.data[0, 0, :]
    code = field.meta["code"]   # H1: real ±1, сырой (до растяжения/несущей)

    spectrum = np.abs(np.fft.fft(sig))
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    order = np.argsort(freqs)

    fig, (ax_code, ax_f) = plt.subplots(2, 1, figsize=(9, 6))
    ax_code.step(np.arange(200), code[:200], where="mid")
    ax_code.set_title(f"ФМн код ±1 (meta['code'], H1) — первые 200 из L={code.shape[0]}")
    ax_code.set_xlabel("k (чип)")
    ax_code.grid(True, alpha=0.3)

    ax_f.plot(freqs[order] / 1e6, spectrum[order])
    ax_f.axvline(_F0 / 1e6, color="r", linestyle="--", alpha=0.5, label="f0")
    ax_f.set_title("ФМн: широкополосный спектр (BPSK, НЕ узкий тон)")
    ax_f.set_xlabel("f, МГц")
    ax_f.legend()
    ax_f.grid(True, alpha=0.3)

    fig.tight_layout()
    writer.write(fig, "phase_code_spectrum.png")
    plt.close(fig)


def _demo_fm_interference(backend: NumpyBackend, writer: FigureWriter) -> None:
    """P4: ЧМ-помеха — спектрограмма (пост. частота с девиацией) + спектр (боковые Бесселя)."""
    fs, n = _FS, 8192
    f_m = fs / 256.0
    spec = WaveformSpec(fs=fs, carrier_hz=_F0, n_samples=n, meta={"beta": 2.0, "f_m": f_m})
    field = FmInterferenceWaveform().render(backend, spec, np.random.default_rng(30))
    sig = field.data[0, 0, :]

    f_spec, t_spec, sxx = spectrogram(sig, fs=fs, nperseg=256, noverlap=224, return_onesided=False)
    order = np.argsort(f_spec)
    spectrum = np.abs(np.fft.fft(sig))
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    forder = np.argsort(freqs)

    fig, (ax_spg, ax_f) = plt.subplots(2, 1, figsize=(9, 7))
    mesh = ax_spg.pcolormesh(t_spec * 1e6, f_spec[order] / 1e6, 10 * np.log10(sxx[order] + 1e-20),
                              shading="auto")
    fig.colorbar(mesh, ax=ax_spg, label="дБ")
    ax_spg.set_title("ЧМ-помеха: спектрограмма (девиация вокруг f0, НЕ линейный чирп)")
    ax_spg.set_xlabel("t, мкс")
    ax_spg.set_ylabel("f, МГц")

    ax_f.plot(freqs[forder] / 1e6, spectrum[forder])
    for k in (-2, -1, 0, 1, 2):
        ax_f.axvline((_F0 + k * f_m) / 1e6, color="r", linestyle="--", alpha=0.3)
    ax_f.set_title("ЧМ-помеха: спектр (несущая + боковые Бесселя f0±k·f_m)")
    ax_f.set_xlabel("f, МГц")
    ax_f.set_xlim((_F0 - 6 * f_m) / 1e6, (_F0 + 6 * f_m) / 1e6)
    ax_f.grid(True, alpha=0.3)

    fig.tight_layout()
    writer.write(fig, "fm_interference.png")
    plt.close(fig)


def _demo_correlator_peak(writer: FigureWriter) -> bool:
    """P4/H1: выход `FMCorrelatorROCm` — острый пик на позиции сдвига (GPU; иначе — no-op).

    Возвращает False (и ничего не пишет), если `.so`/ROCm недоступны — норма на
    Windows/cp312 (та же конвенция, что `_demo_gpu_vs_numpy`, §9 спеки).
    """
    try:
        from common.gpu_context import GPUContextManager
        from core.gpu_libs import loader as gpu_libs
        gpu_libs.require()
        radar = gpu_libs.load("dsp_radar")
        ctx = GPUContextManager.get_rocm()
        if ctx is None:
            raise GpuLibsUnavailableError("ROCmGPUContext недоступен.")
    except (ImportError, GpuLibsUnavailableError) as exc:
        print(f"P4 correlator-демо пропущено (нет .so/ROCm): {exc}")
        return False

    degree = 13
    ref = m_sequence_pow2(degree, seed=1)   # H1: real ±1, H3: pow2-длина под fft_size
    fft_size = ref.shape[0]
    shift = 250
    n_kg = 400

    corr = radar.FMCorrelatorROCm(ctx)
    corr.set_params(fft_size=fft_size, num_shifts=1, num_signals=1, num_output_points=n_kg)
    corr.prepare_reference_from_data(ref)
    signal = np.roll(ref, shift).astype(np.float32)
    peaks = corr.process(signal[None, :])
    row = peaks[0, 0, :]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(row)
    ax.axvline(shift, color="r", linestyle="--", alpha=0.5, label=f"ожидаемый сдвиг d={shift}")
    ax.set_title(f"FMCorrelatorROCm.process(): пик на d={int(np.argmax(row))} "
                 f"(val={row.max():.0f}, ожидали d={shift})")
    ax.set_xlabel("задержка (отсчёты)")
    ax.set_ylabel("корреляция")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    writer.write(fig, "correlator_peak.png")
    plt.close(fig)
    return True


def main() -> None:
    backend = NumpyBackend()
    writer = FigureWriter(_OUT_DIR)

    _demo_cw(backend, writer)
    _demo_lfm(backend, writer)
    _demo_am(backend, writer)
    _demo_window_placement(backend, writer)
    _demo_snr_check(backend, writer)

    print(f"Записано 5 png в {_OUT_DIR}")

    writer_p2 = FigureWriter(_OUT_DIR_P2)
    if _demo_gpu_vs_numpy(writer_p2):
        print(f"Записано 2 png в {_OUT_DIR_P2}")

    writer_p4 = FigureWriter(_OUT_DIR_P4)
    _demo_mseq_autocorr(writer_p4)
    _demo_phase_code_spectrum(backend, writer_p4)
    _demo_fm_interference(backend, writer_p4)
    print(f"Записано 3 png в {_OUT_DIR_P4}")
    if _demo_correlator_peak(writer_p4):
        print(f"Записано correlator_peak.png в {_OUT_DIR_P4}")


if __name__ == "__main__":
    main()
