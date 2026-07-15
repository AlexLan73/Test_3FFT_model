"""Подпакет генерации сигналов (сырое время → SignalField). Реэкспорт публичного API.

P0: `SignalField`/`Modulation`/`AxisKind`, `Waveform`/`WaveformSpec`, `TimeWindow`.
P1: конкретные волны (`CwWaveform`/`LfmWaveform`/`AmWaveform`) + `WaveformFactory`.
P4: `PhaseCodeWaveform`(ФМн) + `FmInterferenceWaveform`(ЧМ-помеха) + `mseq.m_sequence`.
P5: помехи патента + промышленные (`jammers_rf.py`) — `BarrageRfJammer`/`SmspJammer`/
`DrfmRepeaterJammer`/`IndustrialCwJammer`/`ImpulsiveArcJammer`/`VfdHarmonicJammer`.
P5: заполнение куба (`waveform_to_cube.py`) — `WaveformToCube`(Protocol)/`LfmToCube`/
`AmToCube` + `build_lfm_target_volume` (фикс инъекции цели для ЛЧМ-дечирпа, A9-gap1).
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
from .fm import FmInterferenceWaveform
from .heterodyne import dechirp
from .jammers_rf import (
    BarrageRfJammer,
    DrfmRepeaterJammer,
    ImpulsiveArcJammer,
    IndustrialCwJammer,
    SmspJammer,
    VfdHarmonicJammer,
)
from .lfm import LfmWaveform
from .mseq import DEFAULT_DEGREE, m_sequence, m_sequence_pow2
from .phase_code import PhaseCodeWaveform
from .placement import TimeWindow
from .waveform_to_cube import AmToCube, LfmToCube, WaveformToCube, build_lfm_target_volume

__all__ = [
    "SignalField", "Modulation", "AxisKind",
    "Waveform", "WaveformSpec", "GenBackend",
    "TimeWindow",
    "CwWaveform", "LfmWaveform", "AmWaveform", "PhaseCodeWaveform", "FmInterferenceWaveform",
    "BarrageRfJammer", "SmspJammer", "DrfmRepeaterJammer",
    "IndustrialCwJammer", "ImpulsiveArcJammer", "VfdHarmonicJammer",
    "WaveformFactory",
    "m_sequence", "m_sequence_pow2", "DEFAULT_DEGREE",
    "dechirp", "WaveformToCube", "LfmToCube", "AmToCube", "build_lfm_target_volume",
]
