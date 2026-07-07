"""Конфигурация спектрального SNR-эстиматора (VO) и pipeline helpers.

Аналог CfarConfig из cfar_estimator.py (GPUWorkLib/PyPanelAntennas/SNR/).
Порт compute_pipeline_sizes + next_power_of_2 — логика 1-в-1.
"""
from __future__ import annotations

from dataclasses import dataclass

# Default из snr_defaults (плана v4 GPUWorkLib)
_K_TARGET_N_FFT: int = 2048


@dataclass(frozen=True)
class SnrConfig:
    """Параметры спектрального SNR-эстиматора (Value Object, frozen).

    Attributes
    ----------
    target_n_fft : int
        Целевой размер FFT (0 → auto = 2048).
    step_samples : int
        Шаг децимации по времени (0 → auto = ceil(n_samples / target_n_fft)).
    guard_bins : int
        Guard cells с каждой стороны пика (CFAR ref-window).
    ref_bins : int
        Reference cells с каждой стороны guard.
    search_full_spectrum : bool
        True → argmax по [0..n_fft-1]; False → [0..n_fft//2).
    window : str
        Имя оконной функции: "rect" | "hann" | "hamming" | "blackman".
    cfar_estimator : str
        "mean" (CA-CFAR) или "median" (OS-CFAR).
    """

    target_n_fft: int = 2048
    step_samples: int = 0
    guard_bins: int = 3
    ref_bins: int = 8
    search_full_spectrum: bool = True
    window: str = "hann"
    cfar_estimator: str = "mean"


def next_power_of_2(n: int) -> int:
    """Минимальная степень 2 >= n.

    Реплика FFTProcessorROCm::NextPowerOf2 (fft_processor_rocm.cpp:559).
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def compute_pipeline_sizes(
    n_samples: int,
    target_n_fft: int,
    step_samples: int,
) -> tuple[int, int, int]:
    """Вычислить (step_samples, n_actual, n_fft) по auto-правилам SnrConfig.

    Порт compute_pipeline_sizes из cfar_estimator.py (GPUWorkLib):
      1. target_n_fft == 0 → _K_TARGET_N_FFT (2048)
      2. step_samples == 0 → ceil(n_samples / target_n_fft)
      3. n_actual = n_samples // step_samples
      4. n_fft   = next_power_of_2(n_actual)

    Returns
    -------
    (step_samples, n_actual, n_fft)
    """
    if target_n_fft == 0:
        target_n_fft = _K_TARGET_N_FFT
    if step_samples == 0:
        step_samples = (n_samples + target_n_fft - 1) // target_n_fft
    n_actual = n_samples // step_samples
    n_fft = next_power_of_2(n_actual)
    return step_samples, n_actual, n_fft
