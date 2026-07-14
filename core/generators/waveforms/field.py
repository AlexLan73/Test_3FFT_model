"""SignalField — абстрактный носитель сырого времени генератора (VO, §4.0 спеки).

Данные у АМ/ЛЧМ/ФМн/ЧМ формируются одинаково (сырое время на решётке n×n),
различие — в `modulation`/`axes`/`meta`. Не зашиваем «третья ось = дальность»
(это частный случай только ЛЧМ, см. §4.0 spec). Конверторы к нужному представлению
(куб/корреляция/...) — отдельные Strategy, живут в модуле генератора (P1+).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

import numpy as np


class Modulation(Enum):
    """Тип модуляции полезного сигнала/помехи, наполняющей SignalField."""

    AM = "am"
    LFM = "lfm"
    PHASE_CODE = "phase_code"   # ФМн — агильный код 2^n (зонд, гл.6 патента)
    FM_INTERFERENCE = "fm_interference"   # аналоговая ЧМ — сторонний источник/помеха
    CW = "cw"                   # опорный тон (реюз DSP-GPU cw_numpy)
    NOISE = "noise"              # тепловой/аддитивный шум


class AxisKind(Enum):
    """Смысл конкретной оси `SignalField.data` (порядок соответствует `data.shape`)."""

    ANTENNA_X = "antenna_x"
    ANTENNA_Y = "antenna_y"
    FAST_TIME = "fast_time"
    RANGE_BIN = "range_bin"      # появляется после конвертора (ЛЧМ→куб), не на генерации
    CORR_DELAY = "corr_delay"    # ось задержки корреляции (ФМн), после конвертора


@dataclass(frozen=True, eq=False)
class SignalField:
    """Value Object: сырое время + семантика (тип модуляции, оси, fs, несущая, такт).

    ⚠️ `eq=False` (G1, ревью тасков): дефолтный `__eq__` от dataclass сравнил бы `data`
    как numpy-массивы → `truth value of an array is ambiguous`, а `__hash__` упал бы на
    unhashable ndarray. С `eq=False` используется identity-семантика (`__eq__`/`__hash__`
    из `object`) — сравнение и использование как ключа словаря не бросают.
    """

    data: np.ndarray                       # payload, сырое время [nx, ny, n_samples], complex64
    modulation: Modulation
    axes: tuple[AxisKind, ...]             # смысл каждой оси data (len == data.ndim)
    fs: float                              # дискретизация, Гц
    carrier_hz: float                      # несущая/IF, Гц
    tact: int = 0                          # индекс такта (§4.5 — эволюция сцены)
    meta: Mapping[str, float] = field(default_factory=dict)   # ΔF, snr_db, m, f_m, kx, ky …

    def __post_init__(self) -> None:
        if len(self.axes) != self.data.ndim:
            raise ValueError(
                f"len(axes)={len(self.axes)} должен совпадать с data.ndim={self.data.ndim}"
            )
        if self.data.dtype != np.complex64:
            raise ValueError(f"data.dtype должен быть complex64, получено {self.data.dtype}")
        # G2: meta — не мутабельный дефолт + неизменяемость VO. frozen → только через object.__setattr__.
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
