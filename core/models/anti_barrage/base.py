"""Nuller — абстракция углового подавления помехи (Strategy/DIP, H3).

`SubspaceNuller` (ортогональная/косая проекция) и `RobustMvdrNuller` (адаптивный
Capon-луч) реализуют один и тот же контракт `apply(datacube) -> np.ndarray`, но
разными алгоритмами и с разной формой выхода (полный куб vs луч на цель).
`AntiBarragePipeline` зависит от этой абстракции, а не от конкретного класса
(DIP — см. `.claude/rules/05-python-style.md`).

`Protocol` (structural typing), а не ABC — оба класса уже существуют и не
наследуют общий базовый класс; `runtime_checkable` даёт возможность `isinstance`-
проверки при необходимости (по аналогии с `SnrEstimator` в `core/snr/estimator.py`).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Nuller(Protocol):
    """Угловое подавление помехи: сырой куб → очищенные данные.

    Реализации: `SubspaceNuller` (возвращает куб той же формы (nx, ny, K)),
    `RobustMvdrNuller` (возвращает луч на цель, форма (K,)).
    """

    def apply(self, datacube: np.ndarray) -> np.ndarray: ...
