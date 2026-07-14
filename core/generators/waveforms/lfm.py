"""LfmWaveform — линейный чирп (Strategy, §5 спеки).

Формула — `reference.getX_numpy`: **центрированный** чирп (мгновенная частота
`f0 ± fdev/2` вокруг середины окна). Это ровно формула боевого GPU-генератора
`FormSignalGeneratorROCm` (§6.1) → GPU-first: numpy-эталон зеркалит GPU
(решение Alex 2026-07-14 по эскалации P2/G11; `lfm_numpy`-нецентр. — отвергнут).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._pipeline import amplitude_for_snr, render_pipeline
from .base import Waveform, WaveformSpec
from .field import Modulation, SignalField
from .reference import getX_numpy

if TYPE_CHECKING:
    from ..backends.base import GenBackend


class LfmWaveform(Waveform):
    """ЛЧМ: `reference.getX_numpy` (центр. чирп) → окно → n×n → шум (§4.0)."""

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        # norm_val=1.0 — амплитуда как есть (GPU-дефолт norm=1/√2 при обвязке P2 передаём 1.0).
        signal = getX_numpy(
            spec.fs, spec.n_samples, spec.carrier_hz, amplitude, spec.phase, spec.fdev_hz, 1.0
        )
        return render_pipeline(backend, spec, rng, signal, Modulation.LFM)
