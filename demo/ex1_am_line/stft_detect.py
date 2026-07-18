"""ex1-stft — поиск сигнала АНАЛИЗОМ СПЕКТРА по окнам (СТФТ/спектрограмма).

ТЗ Alex (2026-07-18): окно **16 отсчётов** × **Хэмминг** + **16 нулей** (zero-pad ⇒
FFT-32) + **оверлап: шаг 8** (50%). По кадрам строится спектрограмма S[кадр, бин];
сигнал ищется уже В НЕЙ, а не по сырой оси времени.

Почему это сильнее слепой энергетики (denoise C/D): шум в одном бине FFT-32 несёт
~1/16 полной мощности шума (полоса бина ≈ fs/16 с Хэммингом), а узкополосный сигнал
(несущая) собирается в 1-2 бина целиком ⇒ выигрыш ≈ 10·log10(16) ≈ +12 дБ.

Детекция: |S|² бины шума ~ Exp(N̂) (белый шум) ⇒ порог T = −N̂·ln(pfa_cell), N̂ —
медиана всех ячеек спектрограммы / ln2 (реюз `estimate_noise_floor` denoise, DRY).
Кадр «сигнальный», если в нём есть бин > T; кадры сливаются в сегменты (gap_tol),
пересчёт кадр→отсчёт: start = кадр·hop. Гейт −20 дБ (sidelobe blanking, как в denoise).

Запуск:  .venv/Scripts/python.exe demo/ex1_am_line/stft_detect.py
Графики: demo/graphics/ex1_stft/*.png — строка на SNR: [вход | спектрограмма+найденное].
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Конвенция репо: работает форма `python demo/ex1_am_line/stft_detect.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from demo.core import DemoContext, DemoRunner  # noqa: E402
from demo.ex1_am_line.denoise import (  # noqa: E402
    FoundPulse,
    estimate_noise_floor,
    match_counts,
    true_intervals,
)
from demo.ex1_am_line.example import (  # noqa: E402
    CARRIERS,
    FS,
    KIND_AM,
    KIND_RADIO,
    KIND_TITLE,
    N_AXIS,
    SEED,
    SNR_DB,
    _line_signal,
    add_noise_at_snr,
)


@dataclass(frozen=True)
class StftParams:
    """Параметры СТФТ (VO): ТЗ Alex — окно 16 × Хэмминг, +16 нулей, шаг 8."""

    win_len: int = 16          # длина окна анализа, отсчётов
    n_zeros: int = 16          # добавка нулей (zero-pad) ⇒ n_fft = win_len + n_zeros
    hop: int = 8               # шаг кадра (оверлап = win_len − hop = 50%)
    window: str = "hamming"    # оконная функция

    @property
    def n_fft(self) -> int:
        return self.win_len + self.n_zeros

    def make_window(self) -> np.ndarray:
        if self.window == "hamming":
            return np.hamming(self.win_len).astype(np.float64)
        if self.window in ("rect", "none"):
            return np.ones(self.win_len, dtype=np.float64)
        raise ValueError(f"неизвестное окно {self.window!r} (hamming|rect)")


def stft_power(x: np.ndarray, prm: StftParams) -> np.ndarray:
    """Спектрограмма мощности S[кадр, бин] (float64[n_frames, n_fft]). Вход не мутируется.

    Кадр k: x[k·hop : k·hop+win_len] × окно, добитый нулями до n_fft → |FFT|².
    Векторизовано через sliding_window_view (без python-цикла по 500+ кадрам).
    """
    n_frames = 1 + (len(x) - prm.win_len) // prm.hop
    frames = np.lib.stride_tricks.sliding_window_view(x, prm.win_len)[::prm.hop][:n_frames]
    tapered = frames * prm.make_window()[None, :]
    return np.abs(np.fft.fft(tapered, n=prm.n_fft, axis=1)) ** 2


class StftDetector:
    """Детектор по спектрограмме: Exp-порог на ячейку → сигнальные кадры → сегменты.

    `pfa_cell` — вероятность ложной ячейки: бины шума ~ Exp(N̂) ⇒ T = −N̂·ln(pfa_cell).
    N̂ — медиана всех ячеек / ln2 (`estimate_noise_floor`, реюз denoise): сигнал занимает
    малую долю кадров×бинов, медиана его не видит. Кадры с бином>T сливаются с допуском
    `gap_frames`; сегменты короче `min_frames` — отброс; гейт `min_peak_rel_db` — как в
    denoise (звон/боковики слабее сильнейшего сегмента на >=20 дБ — не отдельный сигнал).
    """

    def __init__(self, prm: StftParams | None = None, pfa_cell: float = 1e-4,
                 gap_frames: int = 4, min_frames: int = 2,
                 min_peak_rel_db: float = 20.0) -> None:
        if not (0.0 < pfa_cell < 1.0):
            raise ValueError(f"pfa_cell должен быть в (0,1), получено {pfa_cell}")
        self._prm = prm if prm is not None else StftParams()
        self._pfa_cell = pfa_cell
        self._gap = gap_frames
        self._min_frames = min_frames
        self._min_peak_rel = 10.0 ** (-min_peak_rel_db / 10.0)

    @property
    def params(self) -> StftParams:
        return self._prm

    def spectrogram(self, x: np.ndarray) -> np.ndarray:
        return stft_power(x, self._prm)

    def detect(self, x: np.ndarray) -> tuple[FoundPulse, ...]:
        """Найденные импульсы (в ОТСЧЁТАХ исходной оси). Вход не мутируется."""
        s = self.spectrogram(x)
        n_hat = max(estimate_noise_floor(s, 50.0), 1e-30)
        threshold = -n_hat * np.log(self._pfa_cell)
        frame_hit = (s > threshold).any(axis=1)            # кадр содержит сигнальный бин
        frame_peak = s.max(axis=1)
        idx = np.flatnonzero(frame_hit)
        if idx.size == 0:
            return ()
        prm = self._prm
        segments: list[tuple[int, int]] = []
        seg_start = int(idx[0]); prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i - prev > self._gap:
                segments.append((seg_start, prev))
                seg_start = i
            prev = i
        segments.append((seg_start, prev))
        pulses = [
            FoundPulse(start=a * prm.hop,
                       length=(b - a) * prm.hop + prm.win_len,
                       peak_power=float(frame_peak[a:b + 1].max()))
            for a, b in segments if (b - a + 1) >= self._min_frames
        ]
        if not pulses:
            return ()
        gate = max(pl.peak_power for pl in pulses) * self._min_peak_rel
        return tuple(pl for pl in pulses if pl.peak_power >= gate)

    def carrier_hz_est(self, x: np.ndarray, fs: float = FS) -> float:
        """f̂ = частота argmax-бина усреднённого по кадрам спектра (разрешение fs/n_fft)."""
        mean_spec = self.spectrogram(x).mean(axis=0)
        freqs = np.fft.fftfreq(self._prm.n_fft, d=1.0 / fs)
        return float(freqs[int(np.argmax(mean_spec))])


class Ex1StftDetect(DemoRunner):
    """Поиск сигналов ex1 по спектрограмме (СТФТ 16×Хэмминг +16 нулей, шаг 8)."""

    name = "ex1_stft"
    seed = SEED

    def __init__(self) -> None:
        self._det = StftDetector()
        self._stats: dict[str, Any] = {}

    def visualize(self, ctx: DemoContext) -> dict[str, Figure]:
        figures: dict[str, Figure] = {}
        totals = [0, 0, 0]                                  # [found, expected, false]
        for kind in (KIND_RADIO, KIND_AM):
            for f_c in CARRIERS:
                figures[f"{kind}_fc{int(f_c / 1e6)}_stft"] = self._fig(kind, f_c, totals)
        prm = self._det.params
        self._stats = {
            "stft": f"win={prm.win_len}·{prm.window}+{prm.n_zeros}z, fft={prm.n_fft}, hop={prm.hop}",
            "detect": f"found {totals[0]}/{totals[1]}, false {totals[2]}",
        }
        return figures

    def report_metrics(self, ctx: DemoContext) -> dict[str, Any]:
        return dict(self._stats)

    def _fig(self, kind: str, f_c: float, totals: list[int]) -> Figure:
        n = np.arange(N_AXIS)
        clean = _line_signal(kind, f_c)
        truth = true_intervals(kind, f_c)
        rng = np.random.default_rng(self.seed)
        prm = self._det.params
        freqs_mhz = np.fft.fftshift(np.fft.fftfreq(prm.n_fft, d=1.0 / FS)) / 1e6
        n_rows = len(SNR_DB)
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 1.9 * n_rows), sharex=True)
        for row, snr in enumerate(SNR_DB):
            noisy = add_noise_at_snr(clean, snr, rng)
            tag = "чистый (∞)" if not np.isfinite(snr) else f"SNR={snr:+.0f} дБ"
            ax_in, ax_sp = axes[row]
            ax_in.plot(n, np.abs(noisy), "-", color="tab:gray", lw=0.6)
            ax_in.set_ylabel(tag, fontsize=8)
            ax_in.grid(alpha=0.3)
            ax_in.set_xlim(0, N_AXIS)

            s = self._det.spectrogram(noisy)                # [кадры, бины]
            s_db = 10.0 * np.log10(np.fft.fftshift(s, axes=1).T + 1e-12)
            frame_samples = np.arange(s.shape[0]) * prm.hop
            ax_sp.pcolormesh(frame_samples, freqs_mhz, s_db, cmap="viridis", shading="auto")
            pulses = self._det.detect(noisy)
            found, false = match_counts(pulses, truth)
            totals[0] += found; totals[1] += len(truth); totals[2] += false
            for pl in pulses:                               # найденное — оранжевые рамки
                ax_sp.axvspan(pl.start, pl.end, color="tab:orange", fill=False, lw=1.8)
            for a, b in truth:                              # истина — белые пунктиры
                ax_sp.axvline(a, color="w", lw=0.7, ls="--", alpha=0.8)
                ax_sp.axvline(b, color="w", lw=0.7, ls="--", alpha=0.8)
            f_hat = self._det.carrier_hz_est(noisy)
            ax_sp.set_title(f"f̂={f_hat / 1e6:+.1f} МГц · найдено {len(pulses)} "
                            f"(истинных {found}/{len(truth)}, ложных {false})", fontsize=7)
            ax_sp.set_ylabel("МГц", fontsize=7)
        axes[0, 0].set_title("вход (шум)", fontsize=9)
        axes[0, 1].set_title(f"спектрограмма СТФТ {prm.win_len}×{prm.window}+{prm.n_zeros}z, "
                             f"шаг {prm.hop}\n" + axes[0, 1].get_title(), fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("отсчёт (семпл)")
        fig.suptitle(f"{KIND_TITLE[kind]} · f_m={f_c / 1e6:.0f} МГц · СТФТ-детекция: рамка=найдено, "
                     f"пунктир=истина", y=0.999)
        fig.tight_layout()
        return fig


def main() -> None:
    report = Ex1StftDetect().run()
    print(report)
    for p in report.figures:
        print("  ", p)


if __name__ == "__main__":
    main()
