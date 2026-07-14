"""CwWaveform — опорный тон (Strategy, §5 спеки). Формула — `reference.cw_numpy`."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._pipeline import amplitude_for_snr, render_pipeline
from .base import Waveform, WaveformSpec
from .field import Modulation, SignalField
from .reference import cw_numpy

if TYPE_CHECKING:
    from ..backends.base import GenBackend


class CwWaveform(Waveform):
    """CW-тон: `reference.cw_numpy` → окно → раскладка n×n → шум по SNR (§4.0 пайплайн)."""

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        signal = cw_numpy(spec.fs, spec.n_samples, spec.carrier_hz, amplitude, spec.phase)
        return render_pipeline(backend, spec, rng, signal, Modulation.CW)
