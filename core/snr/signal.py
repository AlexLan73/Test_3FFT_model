"""Генератор CW-сигнала точки с временны́м стробом (PointSignalGenerator).

Модель:
    - комплексный тон длительностью L = round(duration_frac·N), начиная с start,
      размещается в буфере длины N («строб»);
    - остаток окна вне строба — нули;
    - AWGN (комплексный, CN(0, σ²)) добавляется на всю длину N.

Формула тона: A·exp(j·(2π·f_norm·k + φ))
    — идентична make_cw из lfm_signal_generator.py (GPUWorkLib) и
      _SteeredTone._tone (core/generators/sources.py).
    Переписана локально — не тащим приватный класс, завязанный на RangeConfig.

SNR_in — per-sample in-strobe:
    SNR_in = A²/σ²  →  A = √(σ² · 10^(SNR/10))

Позиции строба:
    "left"   : start = 0
    "right"  : start = N − L
    "center" : start = (N − L) // 2
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np

# Допустимые позиции строба
Position = Literal["left", "right", "center"]


class PointSignalGenerator:
    """Генератор строб-тона с AWGN на всю длину буфера.

    Каждый вызов generate() независим; входные параметры не мутируются.
    Библиотечный класс — не печатает, не рисует.

    Usage
    -----
    gen = PointSignalGenerator()
    signal, support = gen.generate(
        n_samples=2048, freq_norm=0.15, snr_in_db=10.0,
        duration_frac=0.5, position="center",
        noise_power=1.0, rng=np.random.default_rng(42),
    )
    """

    def generate(
        self,
        n_samples: int,
        freq_norm: float,
        snr_in_db: float,
        duration_frac: float = 1.0,
        position: Position = "center",
        noise_power: float = 1.0,
        rng: np.random.Generator | None = None,
        phase: float = 0.0,
    ) -> tuple[np.ndarray, slice]:
        """Сгенерировать сигнал точки и вернуть ground-truth строб.

        Parameters
        ----------
        n_samples     : полная длина буфера N
        freq_norm     : нормированная частота тона f_d/f_s ∈ (−0.5, 0.5)
        snr_in_db     : SNR_in = A²/σ² (per-sample в стробе), дБ
        duration_frac : доля длины строба ∈ (0, 1], L = round(frac·N)
        position      : "left" | "right" | "center"
        noise_power   : σ² AWGN (мощность комплексного шума)
        rng           : np.random.Generator; None → default_rng(42)
        phase         : начальная фаза тона, рад

        Returns
        -------
        (signal, support)
            signal  : np.complex64[n_samples] — тон в стробе + AWGN везде
            support : slice — ground-truth позиция строба (start, stop)

        Raises
        ------
        ValueError если duration_frac не в (0, 1].
        ValueError если position неизвестен.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        if not (0.0 < duration_frac <= 1.0):
            raise ValueError(
                f"duration_frac должно быть в (0, 1], получено {duration_frac}"
            )

        # ── строб ────────────────────────────────────────────────────────────
        strobe_len = max(1, round(duration_frac * n_samples))
        if position == "left":
            start = 0
        elif position == "right":
            start = n_samples - strobe_len
        elif position == "center":
            start = (n_samples - strobe_len) // 2
        else:
            raise ValueError(
                f"position должно быть 'left'|'right'|'center', получено {position!r}"
            )
        stop = start + strobe_len
        support = slice(start, stop)

        # ── амплитуда тона: A² = σ² · 10^(SNR/10) ────────────────────────
        amplitude = math.sqrt(noise_power * 10.0 ** (snr_in_db / 10.0))

        # ── тон в стробе (индексы k глобальные, как у make_cw) ──────────────
        # k ∈ [start, stop) — фаза непрерывная по всему буферу
        k_global = np.arange(start, stop, dtype=np.float64)
        theta = 2.0 * math.pi * freq_norm * k_global + phase
        tone = amplitude * (np.cos(theta) + 1j * np.sin(theta))   # complex128

        # ── AWGN на всю длину N ───────────────────────────────────────────────
        sigma = math.sqrt(noise_power / 2.0)
        re = rng.standard_normal(n_samples).astype(np.float64) * sigma
        im = rng.standard_normal(n_samples).astype(np.float64) * sigma
        sig = re + 1j * im  # complex128, CN(0, σ²)

        # ── добавляем тон только в стробе ────────────────────────────────────
        sig[support] += tone

        return sig.astype(np.complex64), support
