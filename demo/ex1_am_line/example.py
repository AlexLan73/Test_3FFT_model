"""ex1 — сырой сигнал на одной временно́й оси (первый тест).

Ось = N=4096 комплексных IQ-отсчётов (семплов), АЦП fs=500 МГц (fast-time, патент гл.2.3).
На ось раздельно (НЕ суммируя) кладём сигналы в прямоугольном окне (`TimeWindow(short)`,
патент §0.3). ДВА типа сигнала для тестов (`radio` и `am`):

  radio — радиоимпульс: несущая f_c в окне; длительность = 4/8/16 периодов НЕСУЩЕЙ.
          Магнитуда |a(t)| = ПРЯМОУГОЛЬНИК (косинусной модуляции нет).
  am    — АМ: a(t)=(1+m·cos(2π f_env t))·exp(j2π f_c t); длительность = 4/8/16 периодов
          НЕСУЩЕЙ f_m (§2 спеки, буквально). Огибающая f_env=f_c/8 — на коротком импульсе
          горбов почти не видно, это ожидаемо (так задано ТЗ).

Несущие f_c = 250 / 100 / 50 МГц. Амплитуда 1. Огибающая АМ: m=0.5, f_env=f_c/8.
Реюз генераторов — `WaveformFactory().create(Modulation.CW|AM)` (единый реестр), формулы
НЕ дублируем.

⚠️ Алиасинг (граничный демо-случай, §0 п.3 спеки): при f_c=250 МГц (= Найквист fs/2) верхняя
боковая АМ `f_c+f_env=281.25 МГц` выходит за Найквист и **заворачивается** (алиасится) в
`281.25-500=-218.75` МГц. Это осознанный граничный кейс, не баг генератора — отмечен в
подписях графиков (`_alias_note`).

На каждый тип: по несущей 2 PNG (_clean / _noise) + 2 сводных (3 сигнала на оси: варианты / шум).
Обёрнуто в `Ex1AmLine(DemoRunner)` (Template Method стенда `demo/core/`, §0 п.2 спеки).

Запуск:  .venv/Scripts/python.exe demo/ex1_am_line/example.py
         .venv/Scripts/python.exe demo/run_all.py
Графики: demo/graphics/ex1_am_line/*.png  (в .gitignore)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Чтобы работала форма `python demo/ex1_am_line/example.py` (конвенция репо, как tests/all_test.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from core.generators.backends import NumpyBackend
from core.generators.waveforms import Modulation, SignalField, WaveformFactory, WaveformSpec
from core.generators.waveforms.placement import TimeWindow
from demo.core import DemoContext, DemoReport, DemoRunner

KIND_RADIO, KIND_AM = "radio", "am"


@dataclass(frozen=True)
class Ex1Params:
    """Параметры ex1 (Value Object) — единый источник констант для функций/класса ниже."""

    fs: float = 500e6                        # АЦП: частота дискретизации (Найквист ±250 МГц)
    n_axis: int = 4096                       # длина временно́й оси (отсчётов/семплов)
    amplitude: float = 1.0
    seed: int = 7
    carriers: tuple[float, ...] = (250e6, 100e6, 50e6)      # несущие f_c
    durations: tuple[int, ...] = (4, 8, 16)                  # периодов несущей[radio]/огибающей[am]
    snr_db: tuple[float, ...] = (np.inf, 20.0, 10.0, 3.0, 0.0, -6.0)   # inf = чистый
    am_m: float = 0.5                        # глубина АМ
    am_env_frac: float = 1.0 / 8.0           # f_env = f_c/8 (огибающая медленнее несущей)
    three_pos: tuple[tuple[float, int], ...] = ((250e6, 300), (100e6, 1600), (50e6, 2900))
    three_units: dict[str, int] = field(default_factory=lambda: {KIND_RADIO: 20, KIND_AM: 6})


_P = Ex1Params()

# ── module-level "constants" (единый источник — Ex1Params, ниже только распаковка) ──
FS = _P.fs
N_AXIS = _P.n_axis
AMPLITUDE = _P.amplitude
SEED = _P.seed
CARRIERS = list(_P.carriers)
DURATIONS = list(_P.durations)          # ← §0 п.1: [4, 8, 16] (было [3,5,7,10])
SNR_DB = list(_P.snr_db)
AM_M = _P.am_m
AM_ENV_FRAC = _P.am_env_frac
THREE_POS = list(_P.three_pos)
THREE_UNITS = dict(_P.three_units)

KIND_TITLE = {KIND_RADIO: "Радиоимпульс", KIND_AM: "АМ-сигнал"}
MAG_TITLE = {KIND_RADIO: "магнитуда |a(t)| — прямоугольник",
             KIND_AM: "магнитуда |a(t)| — волна (горбы)"}
_COLORS = ["tab:blue", "tab:green", "tab:red"]

_FACTORY = WaveformFactory()            # единый вход ко всем генераторам (11 модуляций)


# ── реюз генераторов (не дублируем формулы) ──────────────────────────────────
def env_freq(f_c: float) -> float:
    return abs(f_c) * AM_ENV_FRAC


def dur_samples(kind: str, f_c: float, n_units: int) -> int:
    """Длительность в отсчётах: n_units периодов НЕСУЩЕЙ f_m (§2 спеки, radio и am одинаково)."""
    return int(round(n_units * FS / abs(f_c)))


def _spec_and_modulation(kind: str, f_c: float, n_units: int,
                          t0_samples: int) -> tuple[WaveformSpec, Modulation]:
    d = dur_samples(kind, f_c, n_units)
    window = TimeWindow(kind="short", t0=t0_samples / FS, dur=d / FS)
    if kind == KIND_RADIO:
        return (WaveformSpec(fs=FS, carrier_hz=f_c, n_samples=N_AXIS, amplitude=AMPLITUDE,
                              window=window), Modulation.CW)
    spec = WaveformSpec(fs=FS, carrier_hz=f_c, n_samples=N_AXIS, amplitude=AMPLITUDE,
                        window=window, meta={"m": AM_M, "f_m": env_freq(f_c)})
    return spec, Modulation.AM


def render_field(kind: str, f_c: float, n_units: int, rng: np.random.Generator, *,
                  t0_samples: int = 0) -> SignalField:
    """Полный `SignalField` (реюз `WaveformFactory` — единая точка входа, формулы не дублируем)."""
    spec, mod = _spec_and_modulation(kind, f_c, n_units, t0_samples)
    return _FACTORY.create(mod).render(NumpyBackend(), spec, rng)


def make_pulse(kind: str, f_c: float, n_units: int, *, t0_samples: int = 0) -> np.ndarray:
    """Сигнал в прямоугольном окне (реюз фабрики). Амплитуда 1, без шума, 1D-срез `[0,0,:]`."""
    field_ = render_field(kind, f_c, n_units, np.random.default_rng(SEED), t0_samples=t0_samples)
    return field_.data[0, 0, :].astype(np.complex64)


def make_envelope(kind: str, f_c: float, n_units: int, *, t0_samples: int = 0) -> np.ndarray:
    """Огибающая БЕЗ несущей (Вариант 2 сводных графиков), действительная величина.

    НЕ дублирует формулу АМ ядра (`core/generators/waveforms/am.py`) — это другое
    представление той же сцены для визуализации: реальная огибающая на прямоугольном
    окне (для am: `1+m·cos(2π·f_env·t)`, для radio: сам прямоугольник, т.к. несущая
    без модуляции магнитуды не несёт). f_env/m/окно — те же параметры, что и у `make_pulse`.
    """
    d = dur_samples(kind, f_c, n_units)
    window = TimeWindow(kind="short", t0=t0_samples / FS, dur=d / FS)
    mask = window.mask(N_AXIS, FS)
    if kind == KIND_RADIO:
        return mask.astype(np.float64)
    t = np.arange(N_AXIS, dtype=np.float64) / FS
    envelope = 1.0 + AM_M * np.cos(2.0 * np.pi * env_freq(f_c) * t)
    return np.where(mask, envelope, 0.0)


def add_noise_at_snr(sig: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Комплексный гауссов шум по всей оси под SNR (относительно мощности несущей =1)."""
    if not np.isfinite(snr_db):
        return sig
    noise_power = AMPLITUDE ** 2 / (10.0 ** (snr_db / 10.0))
    return NumpyBackend().add_noise(sig, noise_power, rng)


def _zoom(kind: str, f_c: float) -> int:
    return min(N_AXIS, int(dur_samples(kind, f_c, max(DURATIONS)) * 1.4) + 4)


def _units_label(kind: str, n_units: int, d: int) -> str:
    return f"{n_units} пер.f_m\n({d} отсч)"


def _alias_note(kind: str, f_c: float) -> str:
    """§0 п.3: честная пометка алиасинга для АМ на несущей = Найквист."""
    if kind != KIND_AM:
        return ""
    upper_sideband = abs(f_c) + env_freq(f_c)
    nyquist = FS / 2.0
    if upper_sideband <= nyquist:
        return ""
    wrapped = upper_sideband - FS
    return (f" · ⚠ f_c={f_c/1e6:.0f} МГц=Найквист: верхняя боковая "
            f"{upper_sideband/1e6:.0f} МГц заворачивается (алиасинг) в {wrapped/1e6:.0f} МГц")


# ── графики ─────────────────────────────────────────────────────────────────
# Разнос 3 импульсов по ОДНОЙ оси 4096 (§3.3 спеки: несколько сигналов — разнести по позициям).
PULSE_T0 = (300, 1600, 2900)          # старт-отсчёты для длительностей 4/8/16


def fig_clean(kind: str, f_c: float) -> Figure:
    """ОДНА ось N=4096: все 3 длительности (4/8/16 пер. f_m) разнесены по позициям.

    Сверху Re (несущая), снизу магнитуда |a(t)| — обе панели на полной оси 0..4096.
    """
    n = np.arange(N_AXIS)
    fig, (ax_re, ax_mag) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for n_u, t0, col in zip(DURATIONS, PULSE_T0, _COLORS, strict=True):
        sig = make_pulse(kind, f_c, n_u, t0_samples=t0)
        d = dur_samples(kind, f_c, n_u)
        lbl = f"{n_u} пер. ({d} отсч), t0={t0}"
        ax_re.plot(n, sig.real, "-", color=col, lw=0.8, label=lbl)
        ax_mag.plot(n, np.abs(sig), "-", color=col, lw=1.3, label=lbl)
        ax_mag.fill_between(n, np.abs(sig), color=col, alpha=0.15)
    ax_re.set_title("несущая (Re)")
    ax_mag.set_title(MAG_TITLE[kind])
    for ax in (ax_re, ax_mag):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", ncol=3)
        ax.set_xlim(0, N_AXIS)
    ax_mag.set_xlabel("отсчёт (семпл), ось N=4096, fs=500 МГц")
    fig.suptitle(f"{KIND_TITLE[kind]} · f_m = {f_c/1e6:.0f} МГц · 3 длительности (4/8/16 пер. f_m) "
                 f"на ОДНОЙ оси 4096{_alias_note(kind, f_c)}", y=0.999)
    fig.tight_layout()
    return fig


def _line_signal(kind: str, f_c: float) -> np.ndarray:
    """Та же сцена, что в `fig_clean`: 3 импульса (4/8/16 пер. f_m) на одной оси 4096."""
    clean = np.zeros(N_AXIS, dtype=np.complex64)
    for n_u, t0 in zip(DURATIONS, PULSE_T0, strict=True):
        clean = clean + make_pulse(kind, f_c, n_u, t0_samples=t0)
    return clean


def fig_noise(kind: str, f_c: float) -> Figure:
    """Пара к `fig_clean`: ТА ЖЕ ось 4096 с 3 импульсами, строки — чистый + разные SNR."""
    clean = _line_signal(kind, f_c)
    n = np.arange(N_AXIS)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(len(SNR_DB), 1, figsize=(14, 1.7 * len(SNR_DB)), sharex=True)
    for ax, snr in zip(axes, SNR_DB, strict=True):
        noisy = add_noise_at_snr(clean, snr, rng)
        ax.plot(n, np.abs(noisy), "-", color="tab:red", lw=0.7, label="|сигнал+шум|")
        ax.plot(n, np.abs(clean), "-", color="k", lw=1.2, alpha=0.6, label="|чистый|")
        tag = "чистый (∞)" if not np.isfinite(snr) else f"SNR = {snr:+.0f} дБ"
        ax.set_ylabel(tag, fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, N_AXIS)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("отсчёт (семпл), ось N=4096, fs=500 МГц")
    fig.suptitle(f"{KIND_TITLE[kind]} в шуме · f_m = {f_c/1e6:.0f} МГц · 3 длительности (4/8/16 "
                 f"пер. f_m) на ОДНОЙ оси 4096{_alias_note(kind, f_c)}", y=0.999)
    fig.tight_layout()
    return fig


def fig_three_variants(kind: str) -> Figure:
    """Все 3 несущих на оси 4096 (разнесены), 3 варианта подачи: несущая · огибающая · магнитуда.

    §0 баг-фикс: Вариант 2 — ОГИБАЮЩАЯ БЕЗ несущей (`make_envelope`), Вариант 3 — магнитуда
    |a(t)| (с несущей). Раньше оба рисовали `np.abs(sig)` — были идентичны.
    """
    n = np.arange(N_AXIS)
    n_u = THREE_UNITS[kind]
    sigs = [(f_c, make_pulse(kind, f_c, n_u, t0_samples=t0)) for f_c, t0 in THREE_POS]
    envs = [(f_c, make_envelope(kind, f_c, n_u, t0_samples=t0)) for f_c, t0 in THREE_POS]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for (f_c, sig), (_, env), col in zip(sigs, envs, _COLORS, strict=True):
        lbl = f"f_c={f_c/1e6:.0f} МГц"
        ax1.plot(n, sig.real, "-", color=col, lw=0.7, label=lbl)
        ax2.plot(n, env, "-", color=col, lw=1.3, label=lbl)
        ax2.fill_between(n, env, color=col, alpha=0.13)
        ax3.plot(n, np.abs(sig), "-", color=col, lw=1.3, label=lbl)
    ax1.set_title("Вариант 1 — сигнал с несущей (Re)")
    ax2.set_title("Вариант 2 — огибающая БЕЗ несущей")
    ax3.set_title("Вариант 3 — магнитуда |a(t)| (с несущей)")
    for ax in (ax1, ax2, ax3):
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax3.set_xlabel("отсчёт (семпл),  ось N=4096, fs=500 МГц")
    ax3.set_xlim(0, N_AXIS)
    fig.suptitle(f"{KIND_TITLE[kind]}: три несущих (250/100/50 МГц) на одной оси 4096 — 3 варианта",
                 y=0.999)
    fig.tight_layout()
    return fig


def fig_three_noise(kind: str) -> Figure:
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


class Ex1AmLine(DemoRunner):
    """АМ на одной временно́й оси (Template Method стенда `demo/core/`, §0 п.2 спеки).

    Переопределяет только `build_signal`/`visualize`/`report_metrics` — остальные
    hook'и (build_volume/to_cube/...) остаются заглушками `DemoRunner` (задел под ex2+).
    """

    name = "ex1_am_line"
    seed = SEED

    def build_signal(self, ctx: DemoContext) -> SignalField:
        """Эталонный SignalField для отчёта: АМ, f_c=100 МГц, 8 периодов огибающей."""
        return render_field(KIND_AM, 100e6, 8, ctx.rng)

    def visualize(self, ctx: DemoContext) -> dict[str, Figure]:
        """12 PNG парами: 2 типа (radio/am) × 3 набора f_m (250/100/50) × {clean, noise}.
        Один набор = одна ось 4096 с 3 длительностями (4/8/16 пер. f_m, §2 спеки)."""
        figures: dict[str, Figure] = {}
        for kind in (KIND_RADIO, KIND_AM):
            for f_c in CARRIERS:
                tag = f"{kind}_fc{int(f_c / 1e6)}"
                figures[tag] = fig_clean(kind, f_c)
                figures[f"{tag}_noise"] = fig_noise(kind, f_c)
        return figures

    def report_metrics(self, ctx: DemoContext) -> dict[str, Any]:
        return {
            "n_samples": N_AXIS,
            "carrier_hz": 100e6,
            "m": AM_M,
            "durations": list(DURATIONS),
            "carriers": list(CARRIERS),
        }


def main() -> None:
    report: DemoReport = Ex1AmLine().run()
    print(report)
    for p in report.figures:
        print("  ", p)


if __name__ == "__main__":
    main()
