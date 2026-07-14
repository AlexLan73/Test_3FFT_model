"""Общий хвост пайплайна `Waveform.render` (§4.0 спеки, шаги 2-5) — DRY между cw/lfm/am.

Приватный модуль подпакета (не реэкспортится в `__init__.py`) — деталь реализации,
не публичный API. Шаг 1 (формула конкретной модуляции) остаётся в самой волне;
здесь — общее: окно → раскладка n×n → шум по SNR → упаковка в `SignalField`.

⚠️ R3 (спека §4.6): волна геометрию НЕ решает. `kx`/`ky`/`nx`/`ny` читаются из
`spec.meta` с заглушкой «источник по нормали» (`kx=ky=0.0`) и дефолтом решётки
16×16 (baseline §5.1) — полноценный `SceneModeler` подставит их снаружи позже.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..grid import ArrayGrid
from .base import WaveformSpec
from .field import AxisKind, Modulation, SignalField

if TYPE_CHECKING:
    from ..backends.base import GenBackend

NOISE_POWER: float = 1.0  # эталонная σ² для калибровки амплитуды по snr_db (R5, как core.snr)

_AXES = (AxisKind.ANTENNA_X, AxisKind.ANTENNA_Y, AxisKind.FAST_TIME)


def amplitude_for_snr(spec: WaveformSpec) -> float:
    """`spec.amplitude`, либо `A=√(σ²·10^(snr_db/10))`, если `snr_db` задан.

    R5-математика: IQ-baseband (комплексный) → БЕЗ множителя 2. `σ²=NOISE_POWER` —
    та же эталонная мощность шума, что подаётся в `backend.add_noise` ниже
    (согласованная калибровка, как `core.snr.PointSignalGenerator`).
    """
    if spec.snr_db is None:
        return spec.amplitude
    return math.sqrt(NOISE_POWER * 10.0 ** (spec.snr_db / 10.0))


def grid_from_meta(spec: WaveformSpec) -> tuple[ArrayGrid, float, float]:
    """`(ArrayGrid, kx, ky)` из `spec.meta` — заглушка «по нормали» (R3, P1)."""
    nx = int(spec.meta.get("nx", 16))
    ny = int(spec.meta.get("ny", 16))
    kx = float(spec.meta.get("kx", 0.0))
    ky = float(spec.meta.get("ky", 0.0))
    return ArrayGrid(nx, ny), kx, ky


def render_pipeline(
    backend: GenBackend,
    spec: WaveformSpec,
    rng: np.random.Generator,
    signal: np.ndarray,
    modulation: Modulation,
) -> SignalField:
    """Шаги 2-5 (§4.0): окно → раскладка n×n через steering → шум по SNR → SignalField.

    `signal` — уже посчитанный 1D-сигнал шага 1 (формула конкретной волны, длина
    `spec.n_samples`, complex64).
    """
    mask = spec.window.mask(spec.n_samples, spec.fs)
    windowed = backend.apply_window(signal, mask)

    grid, kx, ky = grid_from_meta(spec)
    steer = grid.steering(kx, ky)
    data = steer[:, :, None] * windowed[None, None, :]

    if spec.snr_db is not None:
        data = backend.add_noise(data, NOISE_POWER, rng)

    return SignalField(
        data=data.astype(np.complex64),
        modulation=modulation,
        axes=_AXES,
        fs=spec.fs,
        carrier_hz=spec.carrier_hz,
        meta=spec.meta,
    )
