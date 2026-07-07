"""core.snr — SNR-эстиматор по временно́му ряду (phase1).

Публичный API:

    SnrConfig              — конфигурация спектрального оценщика (VO, frozen)
    PointSignalGenerator   — генератор строб-тона + AWGN
    SnrResult              — результат оценки (VO, frozen)
    SnrEstimator           — Protocol (интерфейс оценщика)
    SpectrumSnrEstimator   — CA/OS-CFAR по FFT (порт GPUWorkLib)
    StatisticsSnrEstimator — time-domain статистика (формула §TASK_snr_phase1)
    next_power_of_2        — helper: минимальная степень 2 >= n
    compute_pipeline_sizes — helper: (step, n_actual, n_fft) из n_samples
"""
from __future__ import annotations

from .config import SnrConfig, compute_pipeline_sizes, next_power_of_2
from .estimator import SnrEstimator, SnrResult, SpectrumSnrEstimator, StatisticsSnrEstimator
from .signal import PointSignalGenerator

__all__ = [
    "SnrConfig",
    "compute_pipeline_sizes",
    "next_power_of_2",
    "SnrResult",
    "SnrEstimator",
    "SpectrumSnrEstimator",
    "StatisticsSnrEstimator",
    "PointSignalGenerator",
]
