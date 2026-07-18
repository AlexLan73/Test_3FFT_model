"""ex1 — сырой сигнал на одной временно́й оси (первый тест).

Ось = N=4096 комплексных IQ-отсчётов (семплов), АЦП fs=500 МГц (fast-time, патент гл.2.3).
На ось раздельно (НЕ суммируя) кладём сигналы в прямоугольном окне (`TimeWindow(short)`,
патент §0.3). ДВА типа сигнала для тестов (`radio` и `am`):

  radio — радиоимпульс: несущая f_c в окне; длительность = 3/5/7/10 периодов НЕСУЩЕЙ.
          Магнитуда |a(t)| = ПРЯМОУГОЛЬНИК (косинусной модуляции нет).
  am    — АМ: a(t)=(1+m·cos(2π f_env t))·exp(j2π f_c t); длительность = 3/5/7/10 периодов
          ОГИБАЮЩЕЙ. Магнитуда = волна с горбами (НЕ прямоугольник).

Несущие f_c = 250 / 100 / 50 МГц. Амплитуда 1. Огибающая АМ: m=0.5, f_env=f_c/8.
Реюз генераторов — `WaveformFactory().create(Modulation.CW|AM)` (единый реестр), формулы
НЕ дублируем.

На каждый тип: по несущей 2 PNG (_clean / _noise) + 2 сводных (3 сигнала на оси: варианты / шум).

Запуск:  .venv/Scripts/python.exe demo/ex1_am_line/example.py
Графики: demo/graphics/ex1_am_line/*.png  (в .gitignore)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.generators.backends import NumpyBackend
from core.generators.waveforms import Modulation, WaveformFactory, WaveformSpec
from core.generators.waveforms.placement import TimeWindow

# ── параметры оси и сигнала ─────────────────────────────────────────────────
FS = 500e6                       # АЦП: частота дискретизации 500 МГц (Найквист ±250 МГц)
N_AXIS = 4096                    # длина временно́й оси (отсчётов/семплов)
AMPLITUDE = 1.0
SEED = 7

CARRIERS = [250e6, 100e6, 50e6]         # несущие f_c
DURATIONS = [3, 5, 7, 10]               # 4 длительности (периодов несущей [radio] / огибающей [am])
SNR_DB = [np.inf, 20.0, 10.0, 3.0, 0.0, -6.0]   # ∞ = чистый

AM_M = 0.5                       # глубина АМ
AM_ENV_FRAC = 1.0 / 8.0          # f_env = f_c/8 (огибающая медленнее несущей)

KIND_RADIO, KIND_AM = "radio", "am"
KIND_TITLE = {KIND_RADIO: "Радиоимпульс", KIND_AM: "АМ-сигнал"}
MAG_TITLE = {KIND_RADIO: "магнитуда |a(t)| — прямоугольник",
             KIND_AM: "магнитуда |a(t)| — волна (горбы)"}

THREE_POS = [(250e6, 300), (100e6, 1600), (50e6, 2900)]   # (несущая, старт-отсчёт) — разнесены
THREE_UNITS = {KIND_RADIO: 20, KIND_AM: 6}                # длительность сводных (в «единицах» типа)
_COLORS = ["tab:blue", "tab:green", "tab:red"]

_FACTORY = WaveformFactory()            # единый вход ко всем генераторам (11 модуляций)
OUT = Path("demo/graphics/ex1_am_line")


def env_freq(f_c: float) -> float:
    return abs(f_c) * AM_ENV_FRAC


def dur_samples(kind: str, f_c: float, n_units: int) -> int:
    """Длительность в отсчётах: radio → n периодов несущей; am → n периодов огибающей."""
    base = abs(f_c) if kind == KIND_RADIO else env_freq(f_c)
    return int(round(n_units * FS / base))


def make_pulse(kind: str, f_c: float, n_units: int, *, t0_samples: int = 0) -> np.ndarray:
    """Сигнал в прямоугольном окне (реюз фабрики). Амплитуда 1, без шума, 1D-срез `[0,0,:]`."""
    d = dur_samples(kind, f_c, n_units)
    window = TimeWindow(kind="short", t0=t0_samples / FS, dur=d / FS)
    if kind == KIND_RADIO:
        spec = WaveformSpec(fs=FS, carrier_hz=f_c, n_samples=N_AXIS, amplitude=AMPLITUDE,
                            window=window)
        mod = Modulation.CW
    else:
        spec = WaveformSpec(fs=FS, carrier_hz=f_c, n_samples=N_AXIS, amplitude=AMPLITUDE,
                            window=window, meta={"m": AM_M, "f_m": env_freq(f_c)})
        mod = Modulation.AM
    field = _FACTORY.create(mod).render(NumpyBackend(), spec, np.random.default_rng(SEED))
    return field.data[0, 0, :].astype(np.complex64)


def add_noise_at_snr(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Комплексный гауссов шум по всей оси под SNR (относительно мощности несущей =1)."""
    if not np.isfinite(snr_db):
        return sig
    noise_power = AMPLITUDE ** 2 / (10.0 ** (snr_db / 10.0))
    return NumpyBackend().add_noise(sig, noise_power, rng)


def _zoom(kind: str, f_c: float) -> int:
    return min(N_AXIS, int(dur_samples(kind, f_c, max(DURATIONS)) * 1.4) + 4)


def _units_label(kind: str, n_units: int, d: int) -> str:
    what = "пер.несущей" if kind == KIND_RADIO else "пер.огиб."
    return f"{n_units} {what}\n({d} отсч)"


# ── графики ────────────────────────────────────────────────────────────────
def fig_clean(kind: str, f_c: float) -> plt.Figure:
    """Строка на длительность: [несущая Re + огибающая] | [магнитуда |a(t)|]."""
    n = np.arange(N_AXIS)
    xmax = _zoom(kind, f_c)
    mag_color = "tab:red"
    fig, axes = plt.subplots(len(DURATIONS), 2, figsize=(12, 2.0 * len(DURATIONS)), sharex=True)
    for row, n_u in enumerate(DURATIONS):
        sig = make_pulse(kind, f_c, n_u)
        d = dur_samples(kind, f_c, n_u)
        ax_l, ax_r = axes[row]
        ax_l.plot(n, sig.real, "-", color="tab:blue", lw=0.6, label="Re (несущая)")
        ax_l.plot(n, np.abs(sig), "-", color="k", lw=1.4, label="|a(t)| огибающая")
        ax_l.set_ylabel(_units_label(kind, n_u, d), fontsize=8)
        ax_l.grid(alpha=0.3)
        ax_r.plot(n, np.abs(sig), "-", color=mag_color, lw=1.4)
        ax_r.fill_between(n, np.abs(sig), color=mag_color, alpha=0.18)
        ax_r.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 0].set_title("несущая (Re) + огибающая")
    axes[0, 1].set_title(MAG_TITLE[kind])
    for ax in axes[-1]:
        ax.set_xlabel("отсчёт (семпл)")
        ax.set_xlim(0, xmax)
    fig.suptitle(f"{KIND_TITLE[kind]} · несущая f_c = {f_c/1e6:.0f} МГц · ось N=4096, fs=500 МГц · "
                 f"4 длительности", y=0.999)
    fig.tight_layout()
    return fig


def fig_noise(kind: str, f_c: float, n_units: int = 10) -> plt.Figure:
    """Магнитуда (макс. длительность) чистая и при нескольких SNR."""
    clean = make_pulse(kind, f_c, n_units)
    n = np.arange(N_AXIS)
    xmax = _zoom(kind, f_c)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(SNR_DB), 1, figsize=(9, 1.6 * len(SNR_DB)), sharex=True)
    for ax, snr in zip(axes, SNR_DB, strict=True):
        noisy = add_noise_at_snr(clean, snr, rng)
        ax.plot(n, np.abs(noisy), "-", color="tab:red", lw=0.8, label="|a(t)| с шумом")
        ax.plot(n, np.abs(clean), "-", color="k", lw=1.2, alpha=0.6, label="|чистый|")
        tag = "чистый (∞)" if not np.isfinite(snr) else f"SNR = {snr:+.0f} дБ"
        ax.set_ylabel(tag, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("отсчёт (семпл)")
    axes[-1].set_xlim(0, xmax)
    fig.suptitle(f"{KIND_TITLE[kind]} в шуме · f_c = {f_c/1e6:.0f} МГц · {n_units} "
                 f"{'пер.несущей' if kind == KIND_RADIO else 'пер.огиб.'} "
                 f"({dur_samples(kind, f_c, n_units)} отсч)", y=0.999)
    fig.tight_layout()
    return fig


def fig_three_variants(kind: str) -> plt.Figure:
    """Все 3 несущих на оси 4096 (разнесены), 3 варианта подачи: несущая · огибающая · магнитуда."""
    n = np.arange(N_AXIS)
    n_u = THREE_UNITS[kind]
    sigs = [(f_c, make_pulse(kind, f_c, n_u, t0_samples=t0)) for f_c, t0 in THREE_POS]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for (f_c, sig), col in zip(sigs, _COLORS, strict=True):
        lbl = f"f_c={f_c/1e6:.0f} МГц"
        ax1.plot(n, sig.real, "-", color=col, lw=0.7, label=lbl)
        ax2.plot(n, np.abs(sig), "-", color=col, lw=1.3, label=lbl)
        ax2.fill_between(n, np.abs(sig), color=col, alpha=0.13)
        ax3.plot(n, np.abs(sig), "-", color=col, lw=1.3, label=lbl)
    ax1.set_title("Вариант 1 — сигнал с несущей (Re)")
    ax2.set_title("Вариант 2 — просто огибающая")
    ax3.set_title("Вариант 3 — магнитуда |a(t)|")
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax3.set_xlabel("отсчёт (семпл),  ось N=4096, fs=500 МГц")
    ax3.set_xlim(0, N_AXIS)
    fig.suptitle(f"{KIND_TITLE[kind]}: три несущих (250/100/50 МГц) на одной оси 4096 — 3 варианта",
                 y=0.999)
    fig.tight_layout()
    return fig


def fig_three_noise(kind: str) -> plt.Figure:
    """3 сигнала на оси 4096 (разнесены) + шум: магнитуда чистая и при разных SNR."""
    n = np.arange(N_AXIS)
    n_u = THREE_UNITS[kind]
    clean = np.zeros(N_AXIS, dtype=np.complex64)
    for f_c, t0 in THREE_POS:
        clean = clean + make_pulse(kind, f_c, n_u, t0_samples=t0)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(SNR_DB), 1, figsize=(13, 1.7 * len(SNR_DB)), sharex=True)
    for ax, snr in zip(axes, SNR_DB, strict=True):
        noisy = add_noise_at_snr(clean, snr, rng)
        ax.plot(n, np.abs(noisy), "-", color="tab:red", lw=0.7, label="|сигнал+шум|")
        ax.plot(n, np.abs(clean), "-", color="k", lw=1.2, alpha=0.6, label="|чистый|")
        tag = "чистый (∞)" if not np.isfinite(snr) else f"SNR = {snr:+.0f} дБ"
        ax.set_ylabel(tag, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("отсчёт (семпл),  ось N=4096, fs=500 МГц")
    axes[-1].set_xlim(0, N_AXIS)
    fig.suptitle(f"{KIND_TITLE[kind]}: три несущих (250/100/50 МГц) на оси 4096 в шуме — магнитуда",
                 y=0.999)
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for kind in (KIND_RADIO, KIND_AM):
        for f_c in CARRIERS:
            tag = f"{kind}_fc{int(f_c/1e6)}"
            for name, fig in (("clean", fig_clean(kind, f_c)), ("noise", fig_noise(kind, f_c))):
                p = OUT / f"{tag}_{name}.png"
                fig.savefig(p, dpi=120); plt.close(fig); saved.append(str(p))
        for name, fig in (("three_variants", fig_three_variants(kind)),
                          ("three_noise", fig_three_noise(kind))):
            p = OUT / f"{kind}_{name}.png"
            fig.savefig(p, dpi=120); plt.close(fig); saved.append(str(p))
    print(f"Сохранено {len(saved)} PNG в {OUT}:")
    for s in saved:
        print("  ", s)


if __name__ == "__main__":
    main()
