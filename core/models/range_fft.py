"""RangeFft -- глобальный дальностный FFT по оси Z для ЛЧМ-фронтенда (гл.3 §3.2, P5).

Один из двух раздельных FFT патента (A6, SPEC §3: `AngularFft`/`RangeFft`). Дечирп
делает вызывающий код (`waveforms.heterodyne.dechirp`, P5) -- этот класс отвечает
только за: zero-pad -> FFT по последней оси (без окна, rect -- патент §3.2: "цель --
тон на всю длину Z, посегментная оконная обработка не нужна") -> шкала бин->метры.

Шкала -- канонический рецепт словаря `MemoryBank/specs/range_scale_dictionary_2026-07-15.md`
§4: `R(k) = |k_signed| * V1`, `V1 = c*fs/(2*mu*n_fft)`, `mu = fdev/(n/fs)`. Знак: см.
docstring `k_signed_range_axis` -- эмпирически конвенция «мапить по |f|» (словарь §3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_LIGHT = 299_792_458.0  # м/с (совпадает с core.motion.kinematics.C_LIGHT)


def next_pow2(n: int) -> int:
    """Наименьшая степень двойки >= n (n>=1). Локальная копия (см. `config.array_config`,
    приватная там -- дублировать импорт приватного имени из чужого модуля не стали)."""
    p = 1
    while p < n:
        p *= 2
    return p


def k_signed_range_axis(n_fft: int, fs: float, mu: float) -> np.ndarray:
    """`R(k)` для каждого бина сырого (нешифтованного) `np.fft.fft`-вывода длины `n_fft`.

    `k_signed = k` при `k <= n_fft//2`, иначе `k - n_fft` (конвенция `np.fft.fftfreq`).
    `R(k) = |k_signed| * V1` -- «мапить по |f|» (словарь §3: `f_b` отрицательна по
    конструкции дечирпа/знака tau, это не баг, а вопрос конвенции маппинга).
    """
    v1 = C_LIGHT * fs / (2.0 * mu * n_fft)
    k = np.arange(n_fft)
    k_signed = np.where(k <= n_fft // 2, k, k - n_fft)
    return np.abs(k_signed).astype(np.float64) * v1


@dataclass(frozen=True)
class RangeFft:
    """Глобальный дальностный FFT (rect, БЕЗ окна) с zero-pad `n_fft=next_pow2(n)*pad_factor`.

    `pad_factor=4` дефолт -- рекомендация словаря §6 («мерить ширину при N_fft>=4N»,
    иначе грубая сетка недо-разрешает лепесток и ширина занижена иллюзией 1-бина).
    """

    pad_factor: int = 4

    def __post_init__(self) -> None:
        if self.pad_factor < 1:
            raise ValueError(f"pad_factor должен быть >= 1, получено {self.pad_factor}")

    def n_fft_for(self, n: int) -> int:
        return next_pow2(n) * self.pad_factor

    def transform(self, dechirped: np.ndarray, fs: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
        """`(range_domain, r_axis)`. `dechirped` -- (..., n) комплекс; FFT по последней оси."""
        n = dechirped.shape[-1]
        n_fft = self.n_fft_for(n)
        spectrum = np.fft.fft(dechirped, n=n_fft, axis=-1)
        r_axis = k_signed_range_axis(n_fft, fs, mu)
        return spectrum, r_axis
