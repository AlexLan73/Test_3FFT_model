"""Kinematics -- проекция состояния цели в бины куба (Pure Fabrication, P1).

`state -> (az, el, R, vr) -> (kx, ky, range_bin, doppler_phase)`. Геометрия
апертуры (шаг решётки `d`, длина волны `лямбда`) берётся из `ProjectConfig.wave`
(несущая -> лямбда = c/f0; шаг решётки по умолчанию лямбда/2 -- полуволновая
решётка, типовое радарное допущение, переопределяемо явно).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ProjectConfig
from .state import TargetState

C_LIGHT = 299_792_458.0  # м/с


@dataclass(frozen=True)
class KinematicsSample:
    """Один такт проекции: угловые/дальностные величины + бины куба."""

    az: float             # азимут, рад (0 = нормаль решётки)
    el: float              # угол места, рад
    r: float                # наклонная дальность, м
    vr: float               # радиальная скорость, м/с (< 0 -- сближение)
    kx: float                # угловой бин по X (центрированный, как оси SpectralCube)
    ky: float                 # угловой бин по Y
    range_bin: float           # дальностный бин (0-based)
    doppler_phase: float        # накопленная фаза Доплера за такт, рад (заглушка P6)


class Kinematics:
    """Геометрия апертуры + пересчёт state -> бины. Информационный эксперт (GRASP)."""

    def __init__(self, cfg: ProjectConfig, element_spacing_m: float | None = None) -> None:
        if cfg.wave.carrier_hz <= 0:
            raise ValueError("wave.carrier_hz должен быть положительным (нужен для длины волны)")
        self._cfg = cfg
        self._wavelength = C_LIGHT / cfg.wave.carrier_hz
        self._d = element_spacing_m if element_spacing_m is not None else self._wavelength / 2.0
        self._range_resolution = (
            C_LIGHT / (2.0 * cfg.wave.fdev_hz) if cfg.wave.fdev_hz > 0 else 1.0
        )

    @property
    def wavelength_m(self) -> float:
        return self._wavelength

    @property
    def element_spacing_m(self) -> float:
        return self._d

    def project(self, state: TargetState, dt: float = 1.0) -> KinematicsSample:
        """Состояние цели -> угловые/дальностные величины + бины куба."""
        r = float(np.linalg.norm(state.pos))
        if r < 1e-9:
            az = el = 0.0
            vr = 0.0
        else:
            x, y, z = state.pos
            az = float(np.arctan2(x, z))
            el = float(np.arcsin(np.clip(y / r, -1.0, 1.0)))
            vr = float(np.dot(state.pos, state.vel) / r)   # (r.v)/|r|: приближение -> vr<0

        kx = self._cfg.array.nx * self._d * np.sin(az) * np.cos(el) / self._wavelength
        ky = self._cfg.array.ny * self._d * np.sin(el) / self._wavelength
        range_bin = r / self._range_resolution
        # фаза Доплера за такт (задел под FFT по slow-time, реальный этап -- P6/torch)
        doppler_phase = 2.0 * np.pi * (2.0 * vr / self._wavelength) * dt

        return KinematicsSample(
            az=az, el=el, r=r, vr=vr,
            kx=float(kx), ky=float(ky), range_bin=float(range_bin),
            doppler_phase=float(doppler_phase),
        )
