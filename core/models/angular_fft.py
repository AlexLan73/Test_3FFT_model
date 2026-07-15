"""AngularFft -- угловой FFT 16x16 поячеечно, векторизовано по всем бинам дальности (P5).

Второй из двух раздельных FFT патента гл.3 §3.2 (A6, SPEC §3: `AngularFft`/`RangeFft`).
Тот же алгоритм, что угловой блок `Fft3DModel._transform` (fftn по осям (0,1) +
fftshift) -- вынесен отдельной функцией, а НЕ рефактором `Fft3DModel` (A6-обсуждение
в TASK: рефактор `Fft3DModel` возможен, но здесь сознательно не сделан -- класс уже
покрыт тестами, лишний риск регресса не оправдан ради DRY трёх строк; на ревью можно
объединить `Fft3DModel._transform`'s угловой шаг с этой функцией одним общим вызовом).

`axes=(0,1)` в `np.fft.fft2` обрабатывает КАЖДЫЙ срез по 3-й (дальностной) оси
независимо и одновременно -- это и есть "поячеечно на каждом бине дальности" из
патента, без явного python-цикла по бинам (векторизация).
"""
from __future__ import annotations

import numpy as np

from .windows import AxisWindows, RectWindow, WindowFunction


def angular_fft(cube: np.ndarray, aperture_window: WindowFunction | None = None) -> np.ndarray:
    """Угловой FFT по осям (0,1) для каждого среза по оси 2 (range) -- поячеечно.

    `aperture_window` -- тэйпер апертуры, ОДИНАКОВЫЙ по обеим угловым осям (SPEC §5:
    Хэмминг по апертуре обязателен для ЛЧМ-фронтенда); по дальностной оси (t) окна
    НЕТ (`RectWindow`) -- вход уже прошёл дальностный `RangeFft` без окна (rect).
    По умолчанию `aperture_window=None` -> без окна (для сценариев без требования §5).
    """
    window = aperture_window or RectWindow()
    aw = AxisWindows(x=window, y=window, t=RectWindow())
    tapered = aw.apply(cube)
    spectrum = np.fft.fft2(tapered, axes=(0, 1))
    return np.fft.fftshift(spectrum, axes=(0, 1))
