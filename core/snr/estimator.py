"""SNR-эстиматоры по 1D временно́му ряду: спектральный (CFAR) и статистический.

⚠️  ВАЖНО — отличие от core.models.anti_barrage.CaCfarDetector:
    CaCfarDetector работает по range-оси 3D-куба (SpectralCube) и обнаруживает
    цели в угловом пространстве.  Классы здесь — 1D SNR-оценщики по временно́му
    ряду одной антенны/канала (до FFT или после).  Разные задачи, разные входные
    данные, разные выходы.

Классы
------
SnrResult              — frozen VO: snr_db, method, опциональные поля спектра/статистики.
SnrEstimator           — Protocol (интерфейс): estimate(signal, support) → SnrResult.
SpectrumSnrEstimator   — порт estimate_snr_one_antenna (GPUWorkLib cfar_estimator.py).
StatisticsSnrEstimator — time-domain статистика (§ TASK_snr_phase1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .config import SnrConfig, compute_pipeline_sizes

# ── Value Object ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SnrResult:
    """Результат оценки SNR (Value Object, frozen).

    Attributes
    ----------
    snr_db    : оценка SNR, дБ
    method    : "spectrum" | "statistics"
    k_peak    : (спектр) бин FFT-пика
    peak      : (спектр) |X[k_peak]|²
    noise     : (спектр) оценка шумового уровня (CA/OS-CFAR ref-window)
    p_signal  : (статистика) P̂_signal = max(P̂_total − σ̂², ε)
    noise_var : (статистика) σ̂² (оценка дисперсии шума вне строба)
    """

    snr_db: float
    method: str
    k_peak: int | None = None
    peak: float | None = None
    noise: float | None = None
    p_signal: float | None = None
    noise_var: float | None = None


# ── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class SnrEstimator(Protocol):
    """Оценщик SNR по 1D complex сигналу.

    Спектральный оценщик **игнорирует** support.
    Статистический **требует** support; поднимает ValueError если support=None.
    """

    def estimate(
        self,
        signal: np.ndarray,
        support: slice | None = None,
    ) -> SnrResult:
        ...


# ── Window helpers (порт make_window из cfar_estimator.py GPUWorkLib) ────────

def _make_window(name: str, n: int) -> np.ndarray:
    """Оконная функция (float32[n]): rect | hann | hamming | blackman."""
    name = name.lower()
    if name in ("rect", "rectangular", "none"):
        return np.ones(n, dtype=np.float32)
    k = np.arange(n, dtype=np.float64)
    nm1 = n - 1
    if name == "hann":
        w = 0.5 * (1.0 - np.cos(2.0 * math.pi * k / nm1))
    elif name == "hamming":
        w = 0.54 - 0.46 * np.cos(2.0 * math.pi * k / nm1)
    elif name == "blackman":
        w = (0.42 - 0.5 * np.cos(2.0 * math.pi * k / nm1)
             + 0.08 * np.cos(4.0 * math.pi * k / nm1))
    else:
        raise ValueError(
            f"Неизвестное окно: {name!r}. Допустимые: rect, hann, hamming, blackman"
        )
    return w.astype(np.float32)


# ── SpectrumSnrEstimator ─────────────────────────────────────────────────────

class SpectrumSnrEstimator:
    """Спектральная оценка SNR методом CA/OS-CFAR.

    Порт estimate_snr_one_antenna из cfar_estimator.py (GPUWorkLib).
    Pipeline:
      1. Decimation: signal[::step][:n_actual]
      2. Window (hann/rect/…) — после децимации, ДО zero-pad
      3. Zero-pad до n_fft (ближайшая степень 2)
      4. FFT → |X|²
      5. argmax по [0..n_fft] или [0..n_fft//2]
      6. CA-CFAR (mean) / OS-CFAR (median) — wraparound ref-окно
      7. SNR_dB = 10·log10(peak² / noise_est)

    Parameters
    ----------
    config : SnrConfig | None
        None → SnrConfig() (defaults: hann, CA-CFAR, guard=3, ref=8, target_n_fft=2048).
    """

    def __init__(self, config: SnrConfig | None = None) -> None:
        self._cfg = config if config is not None else SnrConfig()

    @property
    def config(self) -> SnrConfig:
        return self._cfg

    def _spectrum(self, signal: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Шаги 1–4 пайплайна: decimation → window → zero-pad → FFT → |X|².

        Общий код для `estimate()` (продолжает CFAR-детектом пика) и `get_mag_sq()`
        (диагностика/графики, без CFAR) — вынесено, чтобы не дублировать формулу.

        Returns
        -------
        tuple[np.ndarray, int, int]
            (mag_sq[n_fft] float32, n_actual, n_fft).
        """
        cfg = self._cfg
        n_samples = len(signal)
        step, n_actual, n_fft = compute_pipeline_sizes(
            n_samples, cfg.target_n_fft, cfg.step_samples
        )

        # 1. Decimation
        decimated = signal[::step][:n_actual].astype(np.complex64)

        # 2. Window (после децимации, до zero-pad — как в WindowedPadDataOp)
        if cfg.window != "rect":
            w = _make_window(cfg.window, n_actual)
            decimated = (decimated * w).astype(np.complex64)

        # 3. Zero-pad
        if n_actual < n_fft:
            padded = np.zeros(n_fft, dtype=np.complex64)
            padded[:n_actual] = decimated
        else:
            padded = decimated[:n_fft]

        # 4. FFT → |X|²
        spectrum = np.fft.fft(padded, n=n_fft)
        mag_sq = (spectrum.real * spectrum.real +
                  spectrum.imag * spectrum.imag).astype(np.float32)
        return mag_sq, n_actual, n_fft

    def estimate(
        self,
        signal: np.ndarray,
        support: slice | None = None,  # игнорируется — спектру ground-truth не нужен
    ) -> SnrResult:
        """Оценить SNR_fft.  support игнорируется.

        Parameters
        ----------
        signal  : complex[n_samples]
        support : игнорируется (для совместимости с Protocol)

        Returns
        -------
        SnrResult с method="spectrum", k_peak, peak, noise заполнены.
        """
        cfg = self._cfg
        mag_sq, _n_actual, n_fft = self._spectrum(signal)

        # 5. argmax
        search_end = n_fft if cfg.search_full_spectrum else n_fft // 2
        k_peak = int(np.argmax(mag_sq[:search_end]))
        peak_sq = float(mag_sq[k_peak])

        # 6. CFAR ref-window с wraparound (порт C++ peak_cfar_kernel)
        ref_values = np.empty(2 * cfg.ref_bins, dtype=np.float32)
        for i in range(cfg.ref_bins):
            offset = cfg.guard_bins + 1 + i
            k_left  = (k_peak - offset) % n_fft
            k_right = (k_peak + offset) % n_fft
            ref_values[2 * i]     = mag_sq[k_left]
            ref_values[2 * i + 1] = mag_sq[k_right]

        noise_est = (float(np.median(ref_values))
                     if cfg.cfar_estimator == "median"
                     else float(ref_values.mean()))
        noise_est = max(noise_est, 1e-30)

        # 7. SNR_dB
        ratio = max(peak_sq / noise_est, 1e-30)
        snr_db = 10.0 * math.log10(ratio)

        return SnrResult(
            snr_db=snr_db,
            method="spectrum",
            k_peak=k_peak,
            peak=peak_sq,
            noise=noise_est,
        )

    def get_mag_sq(self, signal: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Вернуть (mag_sq[n_fft], n_actual, n_fft) для диагностики и графиков.

        Те же шаги 1–4, что и в estimate(), без CFAR.
        """
        return self._spectrum(signal)


# ── StatisticsSnrEstimator ───────────────────────────────────────────────────

class StatisticsSnrEstimator:
    """Time-domain статистическая оценка SNR.

    Не требует SnrConfig (ISP: спектральный конфиг не нужен для time-domain).

    Формулы (§ TASK_snr_phase1, ревью Opus):

        σ̂²       = mean(|x_k|²  по «пустым» отсчётам ВНЕ строба)
                    при center-позиции — оба края объединяются
        P̂_total  = mean(|x_k|²  по стробу support)
        P̂_signal = max(P̂_total − σ̂², ε)          ε = 1e-30
        SNR_stat = 10·log10(P̂_signal / σ̂²)

    ЗАПРЕЩЕНО: наивная P̂_signal = P̂_total (без вычитания σ̂²) — смещение +σ²
    (особенно критично на низком SNR).

    Следствие корректной формулы:
        E[P̂_signal] = A² (не зависит от frac) → SNR_stat ≈ SNR_in.
        Processing gain отсутствует — в отличие от SpectrumSnrEstimator.

    Все редукции (mean, sum) — строго в float64 для числовой устойчивости.
    """

    _EPS: float = 1e-30

    def estimate(
        self,
        signal: np.ndarray,
        support: slice | None = None,
    ) -> SnrResult:
        """Оценить SNR_stat по time-domain статистике.

        Parameters
        ----------
        signal  : complex[n_samples]
        support : slice — ground-truth строб (обязателен)

        Returns
        -------
        SnrResult с method="statistics", p_signal, noise_var заполнены.

        Raises
        ------
        ValueError если support is None.
        """
        if support is None:
            raise ValueError(
                "StatisticsSnrEstimator требует support (ground-truth строб). "
                "Передайте support — slice, возвращённый PointSignalGenerator.generate()."
            )

        n = len(signal)
        # |x_k|² → float64 до mean (числовая устойчивость)
        pow_full = (signal.real.astype(np.float64) ** 2 +
                    signal.imag.astype(np.float64) ** 2)

        # ── мощность в стробе ────────────────────────────────────────────────
        p_total = float(pow_full[support].mean())

        # ── шум вне строба (объединяем левый и правый хвосты) ────────────────
        start = support.start if support.start is not None else 0
        stop  = support.stop  if support.stop  is not None else n

        outside_parts: list[np.ndarray] = []
        if start > 0:
            outside_parts.append(pow_full[:start])
        if stop < n:
            outside_parts.append(pow_full[stop:])

        if not outside_parts:
            # frac=1.0 — строб занимает весь буфер, оценить σ² невозможно по формуле.
            # Клампируем: σ̂² = p_total → p_signal = ε → SNR отрицательный.
            # Это корректное поведение для вырожденного случая.
            sigma2 = max(p_total, self._EPS)
            p_signal = self._EPS
        else:
            outside = (np.concatenate(outside_parts)
                       if len(outside_parts) > 1
                       else outside_parts[0])
            sigma2 = max(float(outside.mean()), self._EPS)
            p_signal = max(p_total - sigma2, self._EPS)

        snr_db = 10.0 * math.log10(p_signal / sigma2)

        return SnrResult(
            snr_db=snr_db,
            method="statistics",
            p_signal=p_signal,
            noise_var=sigma2,
        )
