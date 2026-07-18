"""ex1 — АМ на одной временно́й оси (сырые данные, первый тест).

Ось = N=4096 комплексных IQ-отсчётов (fast-time после дечирпа, патент гл.2.3).
На эту ось раздельно (НЕ суммируя) кладём 4 АМ-сигнала РАЗНОЙ длительности
(`TimeWindow(kind="short")`, патент §0.3: сигнал в любом месте/любой длины) и
считаем МАГНИТУДУ `|a(t)|`. Три несущих: 250 / 100 / 50 МГц.

Сигнал (реюз `AmWaveform`, НЕ дублируем формулу):
    a(t) = (1 + m·cos(2π f_m t))·exp(j 2π f_c t)   — несущая f_c несёт огибающую.

Запуск:  python demo/ex1_am_line/example.py   (дома — .venv/Scripts/python.exe)
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

# единый вход ко ВСЕМ генераторам (11 модуляций): смена сигнала = смена ключа Modulation,
# остальной тракт демо не меняется — задел на ЛЧМ/ФМн/помехи в следующих примерах.
_FACTORY = WaveformFactory()

# ── параметры оси и сигнала ─────────────────────────────────────────────────
FS = 500e6                       # частота дискретизации 500 МГц (Найквист ±250 МГц)
N_AXIS = 4096                    # длина временно́й оси (fast-time), отсчётов
M = 0.5                          # индекс АМ (глубина)
F_M = 2e6                        # частота модуляции огибающей (2 МГц → 250 отсч/период)
AMPLITUDE = 1.0
SEED = 7

CARRIERS = [250e6, 100e6, 50e6]                       # несущие (в полосе ±250 МГц)
DURATIONS = [512, 1024, 2048, 4096]                  # 4 длительности сигнала (отсчётов)

OUT = Path("demo/graphics/ex1_am_line")


def am_on_axis(f_c: float, dur_samples: int, *, t0_samples: int = 0) -> np.ndarray:
    """АМ-сигнал длиной `dur_samples` на оси `N_AXIS` (short-окно), 1D-срез по нормали.

    Вне окна — нули (маска `TimeWindow.short`). Реюз `AmWaveform`; амплитуда 1, без шума.
    """
    spec = WaveformSpec(
        fs=FS, carrier_hz=f_c, n_samples=N_AXIS, amplitude=AMPLITUDE,
        window=TimeWindow(kind="short", t0=t0_samples / FS, dur=dur_samples / FS),
        meta={"m": M, "f_m": F_M},
    )
    field = _FACTORY.create(Modulation.AM).render(NumpyBackend(), spec, np.random.default_rng(SEED))
    return field.data[0, 0, :].astype(np.complex64)


def fig_carrier(f_c: float) -> plt.Figure:
    """Одна несущая: 4 сабплота — 4 длительности на оси 4096, магнитуда |a(t)|."""
    t_ns = np.arange(N_AXIS) / FS * 1e9
    fig, axes = plt.subplots(len(DURATIONS), 1, figsize=(11, 2.0 * len(DURATIONS)), sharex=True)
    for ax, dur in zip(axes, DURATIONS, strict=True):
        sig = am_on_axis(f_c, dur)
        ax.plot(t_ns, sig.real, "-", color="tab:blue", lw=0.4, alpha=0.35, label="Re (несущая)")
        ax.plot(t_ns, np.abs(sig), "-", color="k", lw=1.2, label="|a(t)| магнитуда")
        n_per = dur * abs(f_c) / FS
        ax.set_ylabel(f"dur={dur}\n({dur/N_AXIS*100:.0f}% оси)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.text(0.99, 0.92, f"{n_per:.0f} периодов несущей", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="tab:red")
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("t, нс  (ось N=4096 отсчётов, fs=500 МГц)")
    fig.suptitle(f"АМ на оси 4096 · несущая f_c = {f_c/1e6:.0f} МГц · "
                 f"m={M}, f_m={F_M/1e6:.0f} МГц · 4 длительности (раздельно)", y=0.998)
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for f_c in CARRIERS:
        fig = fig_carrier(f_c)
        p = OUT / f"am_axis4096_fc{int(f_c/1e6)}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(str(p))
    print("Сохранено:")
    for s in saved:
        print("  ", s)


if __name__ == "__main__":
    main()
