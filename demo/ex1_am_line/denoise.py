"""ex1-denoise — найти сигнал в белом шуме СЛЕПЫМ детектором (спека ex1_denoise §0).

Детектор знает ТОЛЬКО: (1) шум белый; (2) сигнал — один из обсуждаемых типов (radio/am).
Несущая НЕИЗВЕСТНА — может быть где угодно в {−250…+250} МГц. t0/длительность — неизвестны.

Два фильтра (Strategy `NoiseFilter`) + общий детекционный хвост:
  C — `SpectralGateFilter`: |FFT|² всей оси → шумовой пол (медиана бинов, Exp-поправка)
      → порог T=−N̂·ln(pfa_bin) → занятые бины (где бы ни были) → маска+запас → iFFT.
  D — `WienerFilter`: N̂ тот же, Ŝ=max(P−N̂,0), H=Ŝ/(Ŝ+N̂) → полосу «находит» неявно.
  Хвост — `PulseDetector`: p=|y|² → OS-CFAR по времени (реюз `OsCfarDetector` core,
      точная Pfa Rohling) → слияние сегментов (gap_tol) → импульсы (start, length).

Оценка несущей f̂ = argmax |FFT(y)|² (несущая доминирует: у АМ m=0.5 → боковые в 4 раза
слабее). ⚠ f_m=250 МГц = Найквист: fftfreq кладёт бин в −250 → сравнение по |f̂|.

Запуск:  .venv/Scripts/python.exe demo/ex1_am_line/denoise.py
Графики: demo/graphics/ex1_denoise/*.png — строка на SNR: [вход | фильтр C | фильтр D].
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Конвенция репо (как tests/all_test.py): работает форма `python demo/ex1_am_line/denoise.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from core.models.tokenizer.cfar import OsCfarDetector  # noqa: E402
from demo.core import DemoContext, DemoRunner  # noqa: E402
from demo.ex1_am_line.example import (  # noqa: E402
    CARRIERS,
    DURATIONS,
    FS,
    KIND_AM,
    KIND_RADIO,
    KIND_TITLE,
    N_AXIS,
    PULSE_T0,
    SEED,
    SNR_DB,
    _line_signal,
    add_noise_at_snr,
    dur_samples,
)


# ── Value Objects ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FoundPulse:
    """Найденный импульс на временно́й оси (VO)."""

    start: int
    length: int
    peak_power: float

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class DenoiseResult:
    """Итог одного прогона фильтр+детектор (VO)."""

    filtered: np.ndarray            # complex64[N] — сигнал после фильтра
    pulses: tuple[FoundPulse, ...]  # найденные импульсы
    carrier_hz_est: float           # f̂ — оценка несущей, Гц (знак как у fftfreq)


# ── шумовой пол спектра (общая математика C и D, DRY) ────────────────────────
def estimate_noise_floor(power_spectrum: np.ndarray, percentile: float = 50.0) -> float:
    """Оценка средней мощности шума на бин по |FFT|² (белый шум ⇒ бины ~ Exp(N)).

    Квантиль p экспоненциального распределения со средним N: q_p = −N·ln(1−p/100)
    ⇒ N̂ = q_p / (−ln(1−p/100)). Дефолт p=50 (медиана): N̂ = медиана/ln2 — устойчиво
    к занятым сигналом бинам (сигнал занимает малую долю оси частот).
    """
    if not (0.0 < percentile < 100.0):
        raise ValueError(f"percentile должен быть в (0,100), получено {percentile}")
    q = float(np.percentile(power_spectrum, percentile))
    return q / (-np.log(1.0 - percentile / 100.0))


class NoiseFilter(Protocol):
    """Strategy: фильтр белого шума. Не мутирует вход, возвращает новый массив."""

    name: str

    def apply(self, x: np.ndarray) -> np.ndarray: ...


class SpectralGateFilter:
    """Вариант C — спектральный поиск занятой полосы (слепой по несущей).

    Белый шум ⇒ |FFT|²-бины ~ Exp(N̂): P(bin > T) = exp(−T/N̂) ⇒ T = −N̂·ln(pfa_bin) —
    честный порог по вероятности ложного бина. Занятые бины расширяем на `margin_bins`
    (края полосы, где мощность уже под порогом) и маскируем спектр.
    """

    name = "C: полоса"

    def __init__(self, pfa_bin: float = 1e-4, margin_bins: int = 8,
                 floor_percentile: float = 50.0) -> None:
        if not (0.0 < pfa_bin < 1.0):
            raise ValueError(f"pfa_bin должен быть в (0,1), получено {pfa_bin}")
        if margin_bins < 0:
            raise ValueError(f"margin_bins должен быть >= 0, получено {margin_bins}")
        self._pfa_bin = pfa_bin
        self._margin = margin_bins
        self._floor_pct = floor_percentile

    def apply(self, x: np.ndarray) -> np.ndarray:
        spec = np.fft.fft(x)
        power = np.abs(spec) ** 2
        n_hat = estimate_noise_floor(power, self._floor_pct)
        threshold = -n_hat * np.log(self._pfa_bin)
        occupied = power > threshold
        if self._margin > 0:  # дилатация маски на margin (свёртка окном единиц)
            kernel = np.ones(2 * self._margin + 1)
            occupied = np.convolve(occupied.astype(np.float64), kernel, mode="same") > 0.0
        return (np.fft.ifft(spec * occupied)).astype(np.complex64)


class WienerFilter:
    """Вариант D — Wiener по оценённому спектру: H = Ŝ/(Ŝ+N̂), Ŝ = max(P−N̂, 0).

    Использует только знание «шум белый» (N̂ — общий `estimate_noise_floor`).
    Полосу сигнала «находит» неявно: H→0 в шумовых бинах, H→1 в сигнальных.
    """

    name = "D: Wiener"

    def __init__(self, floor_percentile: float = 50.0) -> None:
        self._floor_pct = floor_percentile

    def apply(self, x: np.ndarray) -> np.ndarray:
        spec = np.fft.fft(x)
        power = np.abs(spec) ** 2
        n_hat = max(estimate_noise_floor(power, self._floor_pct), 1e-30)
        s_hat = np.maximum(power - n_hat, 0.0)
        h = s_hat / (s_hat + n_hat)
        return (np.fft.ifft(spec * h)).astype(np.complex64)


class PulseDetector:
    """Общий детекционный хвост: p=|y|² → OS-CFAR по времени → сегменты-импульсы.

    Реюз `OsCfarDetector` (core, точная Pfa Rohling) на 1D-оси: `main_half` исключает
    сам импульс из обучающих ячеек (максимальная длительность в ex1 — 160 отсч ⇒
    дефолт 180), train-кольцо = ячейки в (main_half, guard_half]. Соседние отсчёты
    маски сливаются в сегменты с допуском разрыва `gap_tol` (дефолт 32 ≈ длина
    корреляции шума после полосового фильтра; звон у кромки импульса сливается с ним,
    а разнос импульсов ex1 ~1200 — не рискуем склеить разные); короче `min_len` — отброс.

    `min_peak_rel_db` — динамический гейт (sidelobe blanking): полосовая маска/Wiener
    дают «звон» (Гиббс) на −23…−50 дБ ниже импульсов (диагностика 2026-07-18) — CFAR
    честно его видит (звон НЕ шум), но отдельным сигналом он не является. Сегмент
    слабее сильнейшего на >= гейт — отброс. В ex1 импульсы равной амплитуды (1) —
    гейт 20 дБ их не трогает.
    ⚠ Скорость: detect_mask — python-цикл по 4096 ячеек (~0.1 c) — приоритет
    корректность (правило проекта), не оптимизируем преждевременно.
    """

    def __init__(self, pfa: float = 1e-3, main_half: int = 180, guard_half: int = 244,
                 gap_tol: int = 32, min_len: int = 3, min_peak_rel_db: float = 20.0) -> None:
        if gap_tol < 0 or min_len < 1:
            raise ValueError(f"gap_tol>=0, min_len>=1; получено {gap_tol}/{min_len}")
        if min_peak_rel_db <= 0.0:
            raise ValueError(f"min_peak_rel_db должен быть > 0, получено {min_peak_rel_db}")
        self._cfar = OsCfarDetector(pfa=pfa, main_half=main_half, guard_half=guard_half,
                                    percentile=75.0)
        self._gap_tol = gap_tol
        self._min_len = min_len
        self._min_peak_rel = 10.0 ** (-min_peak_rel_db / 10.0)

    def detect(self, y: np.ndarray) -> tuple[FoundPulse, ...]:
        p = (np.abs(y) ** 2).astype(np.float64)
        mask = self._cfar.detect_mask(p)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return ()
        pulses: list[FoundPulse] = []
        seg_start = int(idx[0])
        prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i - prev > self._gap_tol:            # разрыв больше допуска — сегмент закрыт
                pulses.append(self._segment(p, seg_start, prev))
                seg_start = i
            prev = i
        pulses.append(self._segment(p, seg_start, prev))
        kept = [pl for pl in pulses if pl.length >= self._min_len]
        if not kept:
            return ()
        gate = max(pl.peak_power for pl in kept) * self._min_peak_rel
        return tuple(pl for pl in kept if pl.peak_power >= gate)

    @staticmethod
    def _segment(p: np.ndarray, start: int, last: int) -> FoundPulse:
        return FoundPulse(start=start, length=last - start + 1,
                          peak_power=float(p[start:last + 1].max()))


def estimate_carrier_hz(y: np.ndarray, fs: float) -> float:
    """f̂ = частота argmax-бина |FFT(y)|² (несущая доминирует; знак как у fftfreq)."""
    power = np.abs(np.fft.fft(y)) ** 2
    freqs = np.fft.fftfreq(len(y), d=1.0 / fs)
    return float(freqs[int(np.argmax(power))])


def run_denoise(x: np.ndarray, flt: NoiseFilter, det: PulseDetector,
                fs: float = FS) -> DenoiseResult:
    """Связка: фильтр → детектор → оценка несущей. Вход не мутируется."""
    y = flt.apply(x)
    return DenoiseResult(filtered=y, pulses=det.detect(y),
                         carrier_hz_est=estimate_carrier_hz(y, fs))


# ── истина для сверки (генератор знает, детектор — нет) ──────────────────────
def true_intervals(kind: str, f_c: float) -> list[tuple[int, int]]:
    """[(t0, t0+dur)] трёх истинных импульсов набора (4/8/16 пер. f_m)."""
    return [(t0, t0 + dur_samples(kind, f_c, n_u))
            for n_u, t0 in zip(DURATIONS, PULSE_T0, strict=True)]


def match_counts(pulses: tuple[FoundPulse, ...], truth: list[tuple[int, int]]) -> tuple[int, int]:
    """(found, false): found — истинных интервалов, пересечённых хоть одним найденным;
    false — найденных, не пересекающих ни одного истинного."""
    found = sum(1 for a, b in truth if any(pl.start < b and pl.end > a for pl in pulses))
    false = sum(1 for pl in pulses if not any(pl.start < b and pl.end > a for a, b in truth))
    return found, false


# ── пример-обёртка (Template Method стенда) ──────────────────────────────────
class Ex1Denoise(DemoRunner):
    """Слепая детекция сигналов ex1 в белом шуме: фильтры C|D + общий OS-CFAR хвост."""

    name = "ex1_denoise"
    seed = SEED

    def __init__(self) -> None:
        self._filters: tuple[NoiseFilter, ...] = (SpectralGateFilter(), WienerFilter())
        self._detector = PulseDetector()
        self._stats: dict[str, Any] = {}

    def visualize(self, ctx: DemoContext) -> dict[str, Figure]:
        """PNG на (kind, f_m): строки = SNR, колонки = [вход | фильтр C | фильтр D]."""
        figures: dict[str, Figure] = {}
        totals: dict[str, list[int]] = {f.name: [0, 0, 0] for f in self._filters}  # [found, expected, false]
        for kind in (KIND_RADIO, KIND_AM):
            for f_c in CARRIERS:
                fig = self._fig_detect(kind, f_c, totals)
                figures[f"{kind}_fc{int(f_c / 1e6)}_detect"] = fig
        self._stats = {
            "pfa": 1e-3,
            **{name: f"found {v[0]}/{v[1]}, false {v[2]}" for name, v in totals.items()},
        }
        return figures

    def report_metrics(self, ctx: DemoContext) -> dict[str, Any]:
        return dict(self._stats)

    def _fig_detect(self, kind: str, f_c: float, totals: dict[str, list[int]]) -> Figure:
        n = np.arange(N_AXIS)
        clean = _line_signal(kind, f_c)
        truth = true_intervals(kind, f_c)
        rng = np.random.default_rng(self.seed)
        n_rows, n_cols = len(SNR_DB), 1 + len(self._filters)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 1.9 * n_rows),
                                 sharex=True, sharey="row")
        for row, snr in enumerate(SNR_DB):
            noisy = add_noise_at_snr(clean, snr, rng)
            tag = "чистый (∞)" if not np.isfinite(snr) else f"SNR={snr:+.0f} дБ"
            ax_in = axes[row, 0]
            ax_in.plot(n, np.abs(noisy), "-", color="tab:gray", lw=0.6)
            ax_in.set_ylabel(tag, fontsize=8)
            ax_in.grid(alpha=0.3)
            for col, flt in enumerate(self._filters, start=1):
                res = run_denoise(noisy, flt, self._detector)
                found, false = match_counts(res.pulses, truth)
                t = totals[flt.name]
                t[0] += found; t[1] += len(truth); t[2] += false
                ax = axes[row, col]
                ax.plot(n, np.abs(res.filtered), "-", color="tab:red", lw=0.7)
                for pl in res.pulses:                      # найденное — заливка
                    ax.axvspan(pl.start, pl.end, color="tab:orange", alpha=0.28)
                for a, b in truth:                         # истина — пунктиры
                    ax.axvline(a, color="k", lw=0.7, ls="--", alpha=0.55)
                    ax.axvline(b, color="k", lw=0.7, ls="--", alpha=0.55)
                ax.set_title(f"f̂={res.carrier_hz_est / 1e6:+.1f} МГц · найдено {len(res.pulses)}"
                             f" (истинных {found}/{len(truth)}, ложных {false})", fontsize=7)
                ax.grid(alpha=0.3)
        axes[0, 0].set_title("вход (шум)", fontsize=9)
        for col, flt in enumerate(self._filters, start=1):
            axes[0, col].set_title(f"{flt.name}\n" + axes[0, col].get_title(), fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("отсчёт (семпл)")
            ax.set_xlim(0, N_AXIS)
        fig.suptitle(f"{KIND_TITLE[kind]} · f_m={f_c / 1e6:.0f} МГц · слепая детекция: "
                     f"[вход | C: полоса | D: Wiener] · заливка=найдено, пунктир=истина", y=0.999)
        fig.tight_layout()
        return fig


def main() -> None:
    report = Ex1Denoise().run()
    print(report)
    for p in report.figures:
        print("  ", p)


if __name__ == "__main__":
    main()
