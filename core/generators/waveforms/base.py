"""Waveform (ABC/Strategy) + WaveformSpec (VO) — §4.2 спеки.

P0 объявляет абстракцию и чистые данные-спеки. Конкретные модуляции
(`CwWaveform`/`LfmWaveform`/`AmWaveform`/`PhaseCodeWaveform`/...) — P1+.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from .field import SignalField
from .placement import TimeWindow

if TYPE_CHECKING:
    # Настоящий Strategy-протокол бэкенда (§4.3) — `core/generators/backends/base.py` (P1).
    # TYPE_CHECKING + `from __future__ import annotations` (аннотации строковые) —
    # избегаем цикла импорта waveforms↔backends в рантайме.
    from core.generators.backends.base import GenBackend


@dataclass(frozen=True)
class WaveformSpec:
    """Чистые данные для рендера одного `SignalField` (как `LfmParams` в DSP-GPU)."""

    fs: float
    carrier_hz: float
    n_samples: int
    amplitude: float = 1.0
    phase: float = 0.0
    fdev_hz: float = 0.0                    # девиация/полоса ЛЧМ
    snr_db: float | None = None             # R5: калибровка по SNR (None = амплитуда как есть)
    tau_s: float = 0.0                      # задержка
    window: TimeWindow = field(default_factory=lambda: TimeWindow(kind="full"))
    meta: Mapping[str, float] = field(default_factory=dict)   # G10: параметры под тип (AM: m,f_m; …)
    # P4/M1: `add_noise=False` -> амплитуда всё ещё калибрована по `snr_db` (см.
    # `amplitude_for_snr`), но `render_pipeline` шум НЕ подмешивает -- нужно для
    # мульти-цели (N целей суммируются БЕЗ шума, шум добавляется ОДИН раз поверх
    # суммы, не N раз на цель). Дефолт True -- поведение как раньше (совместимость).
    add_noise: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))


class Waveform(ABC):
    """Strategy: конкретная модуляция знает, как синтезировать себя в `SignalField`."""

    @abstractmethod
    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        """Синтезировать SignalField по спеке через бэкенд.

        `rng` передаётся явно (R6: детерминизм) — никаких глобальных `np.random.*`.
        """
