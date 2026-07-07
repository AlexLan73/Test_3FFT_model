"""Источники помех (расширяют SignalSource -- открыты для добавления новых)."""
from __future__ import annotations

import numpy as np

from .sources import SignalSource, _SteeredTone


class DrfmComb(_SteeredTone):
    """Гребёнка ложных целей DRFM: передний фронт + копии ТОЛЬКО позади (вдаль)."""

    def __init__(self, kx, ky, lead_bin: float, spacing: float, count: int,
                 amplitude: float = 1.0, decay: float = 0.85):
        super().__init__(kx, ky)
        self._lead = lead_bin
        self._spacing = spacing
        self._count = count
        self._amp = amplitude
        self._decay = decay

    def contribute(self, grid, rng, rs):
        steer = self._steer(grid)[:, :, None]
        acc = np.zeros((grid.nx, grid.ny, rng.n_real), dtype=complex)
        for i in range(self._count):
            bin_i = self._lead + i * self._spacing      # односторонне: фронт и далее
            amp_i = self._amp * self._decay ** i
            phase = rs.uniform(0, 2 * np.pi)
            tone = self._tone(bin_i, amp_i, phase, rng)
            acc += steer * tone[None, None, :]
        return acc


class BarrageJammer(SignalSource):
    """Заградительная помеха: широкополосный шум с одного направления.

    Когерентна по элементам (один шумовой процесс, наведённый) -> локализована
    по углу, но залита по всей дальностной оси.
    """

    def __init__(self, kx, ky, power: float = 6.0):
        self._kx, self._ky = kx, ky
        self._power = power

    def contribute(self, grid, rng, rs):
        noise = rs.standard_normal(rng.n_real) + 1j * rs.standard_normal(rng.n_real)
        steer = grid.steering(self._kx, self._ky)
        return np.sqrt(self._power) * steer[:, :, None] * noise[None, None, :]


class HamEmitter(SignalSource):
    """Стороннее излучение (радиолюбитель): не коррелирует с нашим ЛЧМ.

    После дерампа CW превращается в чирп -> размаз по всей дальности, под своим углом.
    """

    def __init__(self, kx, ky, amplitude: float = 5.0, chirp_rate=None):
        self._kx, self._ky = kx, ky
        self._amp = amplitude
        self._chirp_rate = chirp_rate

    def contribute(self, grid, rng, rs):
        k = np.arange(rng.n_real)
        beta = self._chirp_rate if self._chirp_rate is not None else 1.0 / rng.n_real
        smear = self._amp * np.exp(1j * np.pi * beta * (k - rng.n_real / 2.0) ** 2)
        steer = grid.steering(self._kx, self._ky)
        return steer[:, :, None] * smear[None, None, :]
