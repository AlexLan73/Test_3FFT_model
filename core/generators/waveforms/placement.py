"""TimeWindow — размещение полезного сигнала по временной оси (VO + Decorator, §4.4 спеки).

Аналог `in_window` из `getX_numpy` (vendored-эталон:
DSP-GPU/DSP/Python/signal_generators/factories.py:87). Применяется декоратором поверх
любого Waveform (P1+) — маска сама по себе не завязана на конкретную модуляцию.
Энергия вне маски = 0 (требование §0.3: сигнал во всём интервале / в части / короткий
в любом месте).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class TimeWindow:
    """Маска размещения сигнала на интервале `[0, n_samples/fs)`.

    - `full`    — весь интервал.
    - `partial` — кусок `[t0, t1)`.
    - `short`   — короткий импульс `[t0, t0+dur)` длиной ровно `round(dur*fs)` отсчётов.
    """

    kind: Literal["full", "partial", "short"]
    t0: float = 0.0
    t1: float | None = None
    dur: float | None = None

    def __post_init__(self) -> None:
        if self.kind == "partial" and self.t1 is None:
            raise ValueError("TimeWindow(kind='partial') требует t1")
        if self.kind == "short" and self.dur is None:
            raise ValueError("TimeWindow(kind='short') требует dur")

    def mask(self, n_samples: int, fs: float) -> np.ndarray:
        """Булева маска длины `n_samples` (True — где сигнал есть)."""
        if self.kind == "full":
            return np.ones(n_samples, dtype=bool)

        m = np.zeros(n_samples, dtype=bool)
        if self.kind == "partial":
            assert self.t1 is not None  # гарантировано __post_init__
            start = max(0, round(self.t0 * fs))
            stop = min(n_samples, round(self.t1 * fs))
            m[start:stop] = True
            return m
        if self.kind == "short":
            assert self.dur is not None  # гарантировано __post_init__
            start = max(0, round(self.t0 * fs))
            length = round(self.dur * fs)
            stop = min(n_samples, start + length)
            m[start:stop] = True
            return m
        raise ValueError(f"неизвестный TimeWindow.kind={self.kind!r}")
