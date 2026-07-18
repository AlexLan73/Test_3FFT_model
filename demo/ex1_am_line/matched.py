"""ex1-matched — согласованная фильтрация ПО МЕСТУ, найденному СТФТ (Вариант A по ТЗ Alex).

Двухэтапная схема (2026-07-18):
  1. **Где сигнал** — многомасштабный СТФТ (реюз `StftDetector`): окна 64/32/16 (+столько же
     нулей, оверлап 50%). Большое окно 64 даёт ≈+18 дБ на бин — ДЛИННЫЙ импульс (>=64 отсч)
     вытаскивается и ниже 0 дБ, где окно 16 уже слепо. ROI = объединение сегментов всех масштабов.
  2. **Какая частота** — по ROI-куску локальный FFT (zero-pad ×4) → argmax →
     **парабола по 3 точкам** (log-мощность): δ = ½(P₋−P₊)/(P₋−2P₀+P₊), f̂=(k+δ)·fs/N —
     точность ≪ бина (напоминание Alex).
  3. **Согласованный фильтр** — банк опор из `WaveformFactory` (формулы НЕ дублируем):
     тип {radio,am} × длительность {4,8,16 пер.} × микросетка частот вокруг f̂.
     Статистика ρ=|corr|²/(σ̂²·‖ref‖²): под H0 (белый шум) ρ~Exp(1) ⇒ порог −ln(pfa).
     Выигрыш когерентного накопления ≈ 10·log10(D): D=160 ⇒ +22 дБ — работает ниже 0 дБ.

Детектор знает семейство сигналов (radio/am, огибающая f_env=f_c/8, m=0.5) — но НЕ знает
несущую (±250 МГц), тип, длительность, t0 конкретного импульса.

Запуск:  .venv/Scripts/python.exe demo/ex1_am_line/matched.py
Графики: demo/graphics/ex1_matched/*.png.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Конвенция репо: работает форма `python demo/ex1_am_line/matched.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from demo.core import DemoContext, DemoRunner  # noqa: E402
from demo.ex1_am_line.denoise import estimate_noise_floor, match_counts, true_intervals  # noqa: E402
from demo.ex1_am_line.example import (  # noqa: E402
    CARRIERS,
    DURATIONS,
    FS,
    KIND_AM,
    KIND_RADIO,
    KIND_TITLE,
    N_AXIS,
    SEED,
    SNR_DB,
    _line_signal,
    add_noise_at_snr,
    dur_samples,
    make_pulse,
)
from demo.ex1_am_line.stft_detect import StftDetector, StftParams  # noqa: E402

# Многомасштабный этап 1: (окно, нули, шаг). 64 — длинные импульсы на низком SNR, 16 — короткие.
STFT_SCALES: tuple[tuple[int, int, int], ...] = ((64, 64, 32), (32, 32, 16), (16, 16, 8))


@dataclass(frozen=True)
class MatchedPulse:
    """Итог согласованной фильтрации одного ROI (VO)."""

    start: int                 # уточнённый t0 (отсчёт)
    length: int                # длительность лучшей опоры (отсчётов)
    kind: str                  # radio | am — лучший тип
    n_units: int               # длительность в периодах (4/8/16)
    carrier_hz: float          # f̂ после параболы
    stat: float                # ρ = |corr|²/(σ̂²·‖ref‖²) в пике

    @property
    def end(self) -> int:
        return self.start + self.length


def parabolic_refine_hz(roi: np.ndarray, fs: float, pad_factor: int = 4) -> float:
    """f̂ по ROI-куску: FFT (zero-pad ×pad_factor) → argmax → парабола 3 точек (log-мощность).

    δ = ½(P₋ − P₊) / (P₋ − 2P₀ + P₊), P в дБ (лог-парабола точнее для оконного пика);
    δ∈[−½,½] бина ⇒ f̂ = (k+δ)·fs/n_fft. На краю спектра (k±1 через wrap) — fftfreq-круг.
    """
    n_fft = 1 << max(4, int(np.ceil(np.log2(len(roi) * pad_factor))))
    p = np.abs(np.fft.fft(roi, n=n_fft)) ** 2
    k = int(np.argmax(p))
    p_m, p_0, p_p = (np.log10(max(p[(k + d) % n_fft], 1e-30)) for d in (-1, 0, 1))
    denom = p_m - 2.0 * p_0 + p_p
    delta = 0.0 if denom >= 0.0 else 0.5 * (p_m - p_p) / denom
    freqs = np.fft.fftfreq(n_fft, d=1.0 / fs)
    step = fs / n_fft
    return float(freqs[k] + delta * step)


def _reference(kind: str, f_c: float, n_units: int, env_shift: int = 0) -> np.ndarray:
    """Опора банка: импульс из `WaveformFactory` (реюз make_pulse), обрезан до D отсчётов.

    `env_shift` (только am) — фаза ОГИБАЮЩЕЙ: формула ядра считает envelope от абсолютного
    времени оси, поэтому импульс с другим t0 стартует с другой фазы огибающей. Опора
    рендерится со сдвигом окна на env_shift и режется от него — та же формула, другая фаза.
    """
    d = max(1, min(dur_samples(kind, f_c, n_units), N_AXIS))
    if env_shift <= 0:
        return make_pulse(kind, f_c, n_units)[:d]
    ref = make_pulse(kind, f_c, n_units, t0_samples=env_shift)[env_shift:env_shift + d]
    return ref


def _env_period_samples(f_c: float) -> int:
    """Период огибающей АМ в отсчётах: f_env=f_c/8 ⇒ T_env = 8·fs/f_c (конвенция ex1)."""
    return max(1, int(round(8.0 * FS / abs(f_c))))


class MatchedFilterBank:
    """Банк согласованных фильтров: тип × длительность × микросетка частот вокруг f̂.

    Статистика ρ=|Σ x·ref*|²/(σ̂²·‖ref‖²): при H0 corr ~ CN(0, σ²‖ref‖²) ⇒ ρ~Exp(1),
    порог ρ > −ln(pfa) — честный по Нейману-Пирсону (фаза неизвестна ⇒ некогерентный |·|).
    Микросетка ±freq_step_hz страхует остаточную ошибку параболы (потеря 3 дБ у D=160
    начинается с ~0.44·fs/D ≈ 1.4 МГц — сетка с шагом 0.5 МГц закрывает с запасом).
    """

    def __init__(self, pfa: float = 1e-4, freq_step_hz: float = 0.5e6,
                 n_freq_side: int = 1) -> None:
        if not (0.0 < pfa < 1.0):
            raise ValueError(f"pfa должен быть в (0,1), получено {pfa}")
        self._pfa = pfa
        self._freq_grid = [i * freq_step_hz for i in range(-n_freq_side, n_freq_side + 1)]

    @property
    def threshold(self) -> float:
        return -float(np.log(self._pfa))

    def best_match(self, x: np.ndarray, roi_start: int, roi_end: int,
                   f_hat: float, noise_power: float) -> MatchedPulse | None:
        """Лучшая опора банка в окрестности ROI [roi_start, roi_end); None — ниже порога.

        Поиск t0 — скольжение опоры по [roi_start−pad, roi_end+pad] (pad=макс. D/2):
        согласованный фильтр сам уточняет позицию, СТФТ давал её грубо (кадр=hop).
        """
        best: MatchedPulse | None = None
        sigma2 = max(noise_power, 1e-30)
        for kind in (KIND_RADIO, KIND_AM):
            for n_units in DURATIONS:
                for df in self._freq_grid:
                    f_c = f_hat + df
                    if abs(f_c) < 1e6 or abs(f_c) > FS / 2:   # вне физичного диапазона
                        continue
                    # am: фаза огибающей неизвестна -> сетка из 4 фаз (0/90/180/270° T_env);
                    # radio: огибающая плоская, фаза не нужна.
                    shifts = ([i * _env_period_samples(f_c) // 4 for i in range(4)]
                              if kind == KIND_AM else [0])
                    refs = [_reference(kind, f_c, n_units, s) for s in shifts]
                    yield_best = self._best_over_refs(x, roi_start, roi_end, refs, sigma2)
                    if yield_best is None:
                        continue
                    k_abs, s, d = yield_best
                    if s >= self.threshold and (best is None or s > best.stat):
                        best = MatchedPulse(start=k_abs, length=d, kind=kind,
                                            n_units=n_units, carrier_hz=f_c, stat=s)
        return best

    @staticmethod
    def _best_over_refs(x: np.ndarray, roi_start: int, roi_end: int,
                        refs: list[np.ndarray], sigma2: float) -> tuple[int, float, int] | None:
        """Максимум ρ по опорам (фазы огибающей): (t0_абс, ρ, D) либо None."""
        best: tuple[int, float, int] | None = None
        for ref in refs:
            d = len(ref)
            pad = max(d // 2, 16)
            a = max(0, roi_start - pad)
            b = min(N_AXIS, roi_end + pad + d)
            seg = x[a:b]
            if len(seg) < d:
                continue
            corr = np.correlate(seg, ref, mode="valid")
            stat = np.abs(corr) ** 2 / (sigma2 * float(np.vdot(ref, ref).real))
            k = int(np.argmax(stat))
            s = float(stat[k])
            if best is None or s > best[1]:
                best = (a + k, s, d)
        return best


class MatchedPipeline:
    """Этапы 1-3 целиком: мультимасштабный СТФТ → парабола f̂ → банк согласованных фильтров."""

    def __init__(self, bank: MatchedFilterBank | None = None) -> None:
        self._detectors = [StftDetector(StftParams(win_len=w, n_zeros=z, hop=h))
                           for w, z, h in STFT_SCALES]
        self._bank = bank if bank is not None else MatchedFilterBank()

    def rois(self, x: np.ndarray) -> list[tuple[int, int]]:
        """ROI = слитое объединение сегментов всех масштабов СТФТ."""
        spans = sorted((pl.start, pl.end) for det in self._detectors for pl in det.detect(x))
        merged: list[tuple[int, int]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return merged

    def find(self, x: np.ndarray) -> tuple[MatchedPulse, ...]:
        """Найденные импульсы (после согласованного фильтра). Вход не мутируется."""
        noise_power = estimate_noise_floor(np.abs(x) ** 2, 50.0)   # |x|²~Exp(σ²) вне сигнала
        found: list[MatchedPulse] = []
        for a, b in self.rois(x):
            f_hat = parabolic_refine_hz(x[a:b], FS)
            mp = self._bank.best_match(x, a, b, f_hat, noise_power)
            if mp is not None:
                found.append(mp)
        return tuple(found)


class Ex1Matched(DemoRunner):
    """Согласованная фильтрация по месту СТФТ: [СТФТ 64/32/16 → парабола → банк опор]."""

    name = "ex1_matched"
    seed = SEED

    def __init__(self) -> None:
        self._pipe = MatchedPipeline()
        self._stats: dict[str, Any] = {}

    def visualize(self, ctx: DemoContext) -> dict[str, Figure]:
        figures: dict[str, Figure] = {}
        totals = [0, 0, 0]                                  # [found, expected, false]
        kind_ok = [0, 0]                                    # [верный тип, всего найденных истинных]
        for kind in (KIND_RADIO, KIND_AM):
            for f_c in CARRIERS:
                figures[f"{kind}_fc{int(f_c / 1e6)}_matched"] = self._fig(kind, f_c, totals, kind_ok)
        self._stats = {
            "pipeline": "stft(64/32/16) -> parabola -> matched bank",
            "detect": f"found {totals[0]}/{totals[1]}, false {totals[2]}",
            "kind_acc": f"{kind_ok[0]}/{kind_ok[1]}",
        }
        return figures

    def report_metrics(self, ctx: DemoContext) -> dict[str, Any]:
        return dict(self._stats)

    def _fig(self, kind: str, f_c: float, totals: list[int], kind_ok: list[int]) -> Figure:
        n = np.arange(N_AXIS)
        clean = _line_signal(kind, f_c)
        truth = true_intervals(kind, f_c)
        rng = np.random.default_rng(self.seed)
        n_rows = len(SNR_DB)
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 1.9 * n_rows), sharex=True)
        for ax, snr in zip(axes, SNR_DB, strict=True):
            noisy = add_noise_at_snr(clean, snr, rng)
            pulses = self._pipe.find(noisy)
            found, false = match_counts(pulses, truth)
            totals[0] += found; totals[1] += len(truth); totals[2] += false
            ax.plot(n, np.abs(noisy), "-", color="tab:gray", lw=0.5)
            labels = []
            for mp in pulses:
                is_true = any(mp.start < b and mp.end > a for a, b in truth)
                if is_true:
                    kind_ok[1] += 1
                    kind_ok[0] += int(mp.kind == kind)
                ax.axvspan(mp.start, mp.end, color="tab:orange", alpha=0.3)
                labels.append(f"{mp.kind}/{mp.n_units}п t0={mp.start} "
                              f"f̂={mp.carrier_hz / 1e6:+.1f}МГц ρ={mp.stat:.0f}")
            for a, b in truth:
                ax.axvline(a, color="k", lw=0.7, ls="--", alpha=0.55)
                ax.axvline(b, color="k", lw=0.7, ls="--", alpha=0.55)
            tag = "чистый (∞)" if not np.isfinite(snr) else f"SNR={snr:+.0f} дБ"
            ax.set_ylabel(tag, fontsize=8)
            ax.set_title(f"найдено {len(pulses)} (истинных {found}/{len(truth)}, ложных {false})"
                         + (" · " + " | ".join(labels) if labels else ""), fontsize=6.5)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, N_AXIS)
        axes[-1].set_xlabel("отсчёт (семпл), ось N=4096, fs=500 МГц")
        fig.suptitle(f"{KIND_TITLE[kind]} · f_m={f_c / 1e6:.0f} МГц · согласованный фильтр по месту "
                     f"СТФТ · заливка=найдено (подпись: тип/длит/t0/f̂/ρ), пунктир=истина", y=0.999)
        fig.tight_layout()
        return fig


def main() -> None:
    report = Ex1Matched().run()
    print(report)
    for p in report.figures:
        print("  ", p)


if __name__ == "__main__":
    main()
