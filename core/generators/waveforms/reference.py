# vendored from DSP-GPU/DSP/Python/signal_generators/factories.py (2026-07-14)
"""ВЕНДОРЕННЫЕ numpy-эталонные формулы генерации (§5 спеки signal_generators_2026-07-13.md).

Портированы **как есть** (не переизобретаем) из DSP-GPU (`cw_numpy`/`lfm_numpy`/
`getX_numpy`, `DSP/Python/signal_generators/factories.py:62,69,79`). Используются как
опорные формулы шага 1 пайплайна `render()` (§4.0) в `waveforms/{cw,lfm}.py` —
дальше волна применяет окно/раскладку n×n/шум единым пайплайном (`_pipeline.py`).
"""
from __future__ import annotations

import numpy as np


def cw_numpy(fs: float, length: int, f0: float,
             amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Эталонный CW сигнал (numpy): `exp(j·(2π·f0·t + phase))`."""
    t = np.arange(length) / fs
    return (amplitude * np.exp(1j * (2 * np.pi * f0 * t + phase))).astype(np.complex64)


def lfm_numpy(fs: float, length: int, f_start: float, f_end: float,
              amplitude: float = 1.0) -> np.ndarray:
    """Эталонный ЛЧМ сигнал (numpy): линейный чирп `f_start → f_end` за `length/fs`."""
    t = np.arange(length) / fs
    duration = length / fs
    chirp_rate = (f_end - f_start) / duration
    phase = 2 * np.pi * (f_start * t + 0.5 * chirp_rate * t ** 2)
    return (amplitude * np.exp(1j * phase)).astype(np.complex64)


def getX_numpy(fs: float, points: int, f0: float, amplitude: float,  # noqa: N802 — вендоренное имя
               phase: float, fdev: float, norm_val: float, tau: float = 0.0) -> np.ndarray:
    """CPU reference FormSignal (формула getX без шума) — ЛЧМ с оконной маской `in_window`.

    Опорная формула размещения из P0 `TimeWindow` (`in_window` — прообраз `TimeWindow.mask`).
    """
    dt = 1.0 / fs
    ti = points * dt
    t = np.arange(points, dtype=np.float64) * dt + tau

    in_window = (t >= 0.0) & (t <= ti - dt)
    t_centered = t - ti / 2.0
    ph = 2.0 * np.pi * f0 * t + np.pi * fdev / ti * (t_centered ** 2) + phase

    x = amplitude * norm_val * np.exp(1j * ph)
    x[~in_window] = 0.0
    return x.astype(np.complex64)
