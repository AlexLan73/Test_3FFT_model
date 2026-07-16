"""Value Objects токенизатора (гл.4 §4.7, гл.4-бис §4-бис.3, TASK §2).

`SquareToken` (`core/graphics/square_view.py`) НЕ трогаем -- он остаётся контрольным
видом (reduce+argmax). `SliceToken`/`RangeVerdict` -- новые VO полного OS-CFAR-детектора.
"""
from __future__ import annotations

from dataclasses import dataclass

from .features import FeatureVector

# Метки прохода 1 (гл.4 §4.8) -- гейт, не арбитр.
SliceLabel = str    # "noise" | "source" | "smeared"

# Виды прохода 2 (гл.4 §4.9).
RangeKind = str     # "target" | "comb" | "barrage"
TARGET = "target"
COMB = "comb"
BARRAGE = "barrage"


@dataclass(frozen=True)
class PeakInfo:
    """Один угловой пик в срезе/окне (кросс-язычно, под C++/msgpack)."""

    kx: float
    ky: float
    amp: float
    edge: float     # кромка: |A|(r+1) - |A|(r-1) в этой угловой ячейке (0.0 на краю куба)


@dataclass(frozen=True)
class SliceToken:
    """Токен на срез/окно дальности `r` (гл.4 §4.7, TASK §2).

    `r`       -- индекс бина дальности (для L=1 -- сам бин; для L>1 -- начало окна `k_z`).
    `peaks`   -- до 5 угловых пиков (гл.4 §4.6, `n_peaks` = len(peaks)).
    `f`       -- 6 признаков (гл.4 §4.5).
    `label`   -- "noise"|"source"|"smeared" (проход 1, гл.4 §4.8).
    `score`   -- уверенность триажа (не арбитр -- гейт).
    """

    r: int
    peaks: tuple[PeakInfo, ...]
    f: FeatureVector
    label: SliceLabel
    score: float

    @property
    def n_peaks(self) -> int:
        return len(self.peaks)


@dataclass(frozen=True)
class RangeVerdict:
    """Результат прохода 2 (гл.4 §4.9) -- сборка `SliceToken` по дальности под одним углом."""

    kx: float
    ky: float
    kind: RangeKind
    lead_r: int
    period_dr: float | None = None
