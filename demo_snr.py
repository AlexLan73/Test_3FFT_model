"""Демо SNR-эстиматора: генерация сигнала точки + оценка SNR (спектр + статистика).

Три графика (out/figures/snr_*.png):
    snr_curve.png     — измеренный SNR (спектр + статистика) vs SNR_in + теория.
    snr_boundary.png  — краевые условия: SNR vs duration_frac (строб у края/центра).
    snr_spectrum.png  — |X|² одной реализации с отметкой пика и CFAR ref-окна.

Общие параметры: N=2048 → step=1 (чистая физика строба), freq_norm=0.15, window=Hann.

⚠️ Статистический оценщик требует «пустую» зону вне строба → свип (а) и статистику
   гоняем при frac=0.5 (при frac=1 σ̂² не оценить, оценка вырождается).

Запуск:  python demo_snr.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.snr import (
    PointSignalGenerator,
    SnrConfig,
    SpectrumSnrEstimator,
    StatisticsSnrEstimator,
    compute_pipeline_sizes,
)

OUT = Path("out/figures")

N = 2048
FREQ = 0.15
NOISE_POWER = 1.0
SWEEP_FRAC = 0.5   # статистика требует frac<1 (нужна шумовая зона)


def _theory_spectrum_db(snr_in_db: float, n_actual: int, frac: float) -> float:
    """Строб-поправленная теория: SNR_in + 10log10(n_actual) + 20log10(frac)."""
    return snr_in_db + 10.0 * math.log10(n_actual) + 20.0 * math.log10(frac)


def plot_curve(gen: PointSignalGenerator, spec: SpectrumSnrEstimator,
               stat: StatisticsSnrEstimator, n_trials: int = 20) -> None:
    """(а) Кривая: измеренный SNR (спектр+статистика) vs SNR_in при frac=0.5."""
    _, n_actual, _ = compute_pipeline_sizes(N, spec.config.target_n_fft, spec.config.step_samples)
    snr_in_list = [0, 5, 10, 15, 20, 25, 30, 35, 40]

    spec_mean, spec_std, stat_mean, stat_std = [], [], [], []
    for snr_in in snr_in_list:
        sp, st = [], []
        for seed in range(n_trials):
            sig, sup = gen.generate(N, FREQ, float(snr_in), SWEEP_FRAC, "center",
                                    NOISE_POWER, np.random.default_rng(seed))
            sp.append(spec.estimate(sig).snr_db)
            st.append(stat.estimate(sig, sup).snr_db)
        spec_mean.append(float(np.mean(sp)))
        spec_std.append(float(np.std(sp)))
        stat_mean.append(float(np.mean(st)))
        stat_std.append(float(np.std(st)))

    theory = [_theory_spectrum_db(s, n_actual, SWEEP_FRAC) for s in snr_in_list]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(snr_in_list, spec_mean, yerr=spec_std, fmt="o-", color="#1f77b4",
                capsize=3, label="спектр (CA-CFAR)")
    ax.errorbar(snr_in_list, stat_mean, yerr=stat_std, fmt="s-", color="#2ca02c",
                capsize=3, label="статистика (P̂_signal/σ̂²)")
    ax.plot(snr_in_list, theory, "k--", alpha=0.7,
            label=f"теория спектра = SNR_in + 10log10({n_actual}) + 20log10({SWEEP_FRAC})")
    ax.plot(snr_in_list, snr_in_list, ":", color="gray", alpha=0.8,
            label="теория статистики = SNR_in (нет processing gain)")
    ax.set_xlabel("SNR_in (дБ)")
    ax.set_ylabel("измеренный SNR (дБ)")
    ax.set_title(f"SNR: спектр vs статистика (N={N}, frac={SWEEP_FRAC}, Hann, {n_trials} реализаций)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "snr_curve.png", dpi=110)
    plt.close(fig)
    print(f"  spec@20дБ={spec_mean[4]:.1f}, stat@20дБ={stat_mean[4]:.1f} (SNR_in=20)")


def plot_boundary(gen: PointSignalGenerator, spec: SpectrumSnrEstimator,
                  stat: StatisticsSnrEstimator, n_trials: int = 20) -> None:
    """(б) Краевые условия: SNR vs frac. Спектр — 3 позиции; статистика — 1 линия+band."""
    snr_in = 5.0
    fracs = [0.3, 0.2, 0.1, 0.05]
    positions = ["left", "center", "right"]
    pos_color = {"left": "#d62728", "center": "#1f77b4", "right": "#ff7f0e"}

    spec_by_pos: dict[str, list[float]] = {p: [] for p in positions}
    stat_mean, stat_std = [], []
    for frac in fracs:
        for pos in positions:
            vals = [spec.estimate(gen.generate(N, FREQ, snr_in, frac, pos, NOISE_POWER,
                                               np.random.default_rng(s))[0]).snr_db
                    for s in range(n_trials)]
            spec_by_pos[pos].append(float(np.mean(vals)))
        sv = [stat.estimate(*gen.generate(N, FREQ, snr_in, frac, "center", NOISE_POWER,
                                          np.random.default_rng(s))).snr_db
              for s in range(n_trials)]
        stat_mean.append(float(np.mean(sv)))
        stat_std.append(float(np.std(sv)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for pos in positions:
        ax.plot(fracs, spec_by_pos[pos], "o-", color=pos_color[pos],
                label=f"спектр [{pos}]")
    ax.errorbar(fracs, stat_mean, yerr=stat_std, fmt="s--", color="#2ca02c",
                capsize=3, label="статистика (center, ±σ)")
    ax.axhline(snr_in, color="gray", ls=":", alpha=0.7, label=f"SNR_in={snr_in:.0f} дБ")
    ax.set_xlabel("duration_frac (доля длины строба)")
    ax.set_ylabel("измеренный SNR (дБ)")
    ax.set_title("Краевые условия: короткий строб у края/центра\n"
                 "спектр падает с frac и зависит от позиции (тапер Hann); статистика ≈ SNR_in")
    ax.invert_xaxis()
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "snr_boundary.png", dpi=110)
    plt.close(fig)
    print(f"  спектр center: frac0.3→{spec_by_pos['center'][0]:.1f}дБ, "
          f"frac0.05→{spec_by_pos['center'][-1]:.1f}дБ; статистика≈{np.mean(stat_mean):.1f}дБ")


def plot_spectrum(gen: PointSignalGenerator, spec: SpectrumSnrEstimator) -> None:
    """(в) |X|² одной реализации с отметкой пика и CFAR ref-окна."""
    sig, _ = gen.generate(N, FREQ, 20.0, 1.0, "center", NOISE_POWER, np.random.default_rng(7))
    mag_sq, _n_actual, n_fft = spec.get_mag_sq(sig)
    res = spec.estimate(sig)
    k_peak = res.k_peak if res.k_peak is not None else int(np.argmax(mag_sq))

    db = 10.0 * np.log10(mag_sq + 1e-12)
    db -= db.max()
    cfg = spec.config

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(db, color="#1f77b4", lw=0.8)
    ax.axvline(k_peak, color="red", ls="--", label=f"пик k={k_peak}")
    for i in range(cfg.ref_bins):
        off = cfg.guard_bins + 1 + i
        for kref in ((k_peak - off) % n_fft, (k_peak + off) % n_fft):
            ax.axvline(kref, color="green", alpha=0.25)
    ax.axvline(-1, color="green", alpha=0.25, label=f"CFAR ref (guard={cfg.guard_bins}, ref={cfg.ref_bins})")
    ax.set_xlim(max(0, k_peak - 60), min(n_fft, k_peak + 60))
    ax.set_xlabel("бин FFT")
    ax.set_ylabel("|X|² (дБ отн. пика)")
    ax.set_title(f"Спектр одной реализации: SNR_fft={res.snr_db:.1f} дБ "
                 f"(freq_norm={FREQ}, n_fft={n_fft}, Hann)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "snr_spectrum.png", dpi=110)
    plt.close(fig)
    print(f"  спектр демо: пик k={k_peak}, SNR_fft={res.snr_db:.1f} дБ")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gen = PointSignalGenerator()
    spec = SpectrumSnrEstimator(SnrConfig())
    stat = StatisticsSnrEstimator()

    print("=" * 68)
    print("  SNR-эстиматор · demo (спектр CA-CFAR + статистика time-domain)")
    print("=" * 68)
    plot_curve(gen, spec, stat)
    plot_boundary(gen, spec, stat)
    plot_spectrum(gen, spec)
    print("  PNG:", ", ".join(str(OUT / n) for n in
                              ("snr_curve.png", "snr_boundary.png", "snr_spectrum.png")))
    print("=" * 68)


if __name__ == "__main__":
    main()
