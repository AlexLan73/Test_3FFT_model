"""Подпакет генерации сигналов (сырое время → SignalField). Реэкспорт публичного API.

P0: `SignalField`/`Modulation`/`AxisKind`, `Waveform`/`WaveformSpec`, `TimeWindow`.
P1: конкретные волны (`CwWaveform`/`LfmWaveform`/`AmWaveform`) + `WaveformFactory`.
`GenBackend` — настоящий протокол теперь в `core.generators.backends` (P1); реэкспорт
здесь оставлен для обратной совместимости с P0-импортами.
"""
from __future__ import annotations

from ..backends import GenBackend
from .am import AmWaveform
from .base import Waveform, WaveformSpec
from .cw import CwWaveform
from .factory import WaveformFactory
from .field import AxisKind, Modulation, SignalField
from .lfm import LfmWaveform
from .placement import TimeWindow

__all__ = [
    "SignalField", "Modulation", "AxisKind",
    "Waveform", "WaveformSpec", "GenBackend",
    "TimeWindow",
    "CwWaveform", "LfmWaveform", "AmWaveform",
    "WaveformFactory",
]
