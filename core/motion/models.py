"""MotionModel -- стратегии движения цели (Strategy, P1).

Все модели продвигают `TargetState` на один такт `dt` (секунды). Аэро-лимиты
(`max_turn_rate`, `max_accel`) клипуются на каждом такте -- без рывков (SPEC:
«движение без резких поворотов»).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .state import TargetState

_EPS = 1e-9


class MotionModel(Protocol):
    """Strategy: закон движения. `rng` -- источник случайности для стохастических моделей
    (детерминированные модели его игнорируют)."""

    def propagate(self, state: TargetState, dt: float, rng: np.random.Generator) -> TargetState: ...


def _direction_and_speed(vel: np.ndarray) -> tuple[np.ndarray, float]:
    """Единичный вектор направления + модуль скорости. Вырожденный случай -- вперёд по Z."""
    speed = float(np.linalg.norm(vel))
    if speed < _EPS:
        return np.array([0.0, 0.0, 1.0]), 0.0
    return vel / speed, speed


def _az_el_from_direction(direction: np.ndarray) -> tuple[float, float]:
    """Направление (единичный вектор) -> (азимут, угол места), рад. Z -- вперёд, Y -- вверх."""
    az = float(np.arctan2(direction[0], direction[2]))
    el = float(np.arcsin(np.clip(direction[1], -1.0, 1.0)))
    return az, el


def _direction_from_az_el(az: float, el: float) -> np.ndarray:
    return np.array([
        np.sin(az) * np.cos(el),
        np.sin(el),
        np.cos(az) * np.cos(el),
    ])


def _integrate_state(state: TargetState, new_vel: np.ndarray, dt: float) -> TargetState:
    """Semi-implicit интегрирование: ускорение из перепада скорости, позиция -- по нему.

    Общая формула для `MarkovDrift`/`CoordinatedTurn`/`WeavingManeuver`: модель сама
    решает, какой должна стать скорость на такте (`new_vel`), а ускорение и позиция
    получаются одинаково:
        acc     = (new_vel - state.vel) / dt   (0, если dt слишком мал)
        new_pos = state.pos + state.vel * dt + 0.5 * acc * dt**2
    """
    acc = (new_vel - state.vel) / dt if dt > _EPS else np.zeros(3)
    new_pos = state.pos + state.vel * dt + 0.5 * acc * dt * dt
    return state.evolved(pos=new_pos, vel=new_vel, acc=acc)


@dataclass
class ConstantVelocity:
    """Прямолинейное равномерное движение: `pos += vel * dt`."""

    def propagate(self, state: TargetState, dt: float, rng: np.random.Generator) -> TargetState:
        new_pos = state.pos + state.vel * dt
        return state.evolved(pos=new_pos, vel=state.vel, acc=np.zeros(3))


@dataclass
class MarkovDrift:
    """Ограниченное случайное блуждание курса/скорости (OU-подобный процесс).

    Малые приращения на такт, клип по аэро-лимитам -- траектория гладкая, без
    рывков (см. SPEC/TASK). `max_turn_rate` -- рад/такт (уже в единицах dt,
    не рад/с, чтобы не плодить лишний параметр в P1); `max_accel` -- м/с^2.
    """

    max_turn_rate: float = 0.03       # клип на |dAz|, |dEl| за такт (рад)
    max_accel: float = 0.6            # клип на |dскорость/dt| (м/с^2)
    heading_noise_std: float = 0.012  # std случайного приращения курса (рад/такт)
    speed_noise_std: float = 0.15     # std случайного приращения скорости (м/с/такт)

    def propagate(self, state: TargetState, dt: float, rng: np.random.Generator) -> TargetState:
        direction, speed = _direction_and_speed(state.vel)
        az, el = _az_el_from_direction(direction)

        d_az = float(np.clip(rng.normal(0.0, self.heading_noise_std),
                              -self.max_turn_rate, self.max_turn_rate))
        d_el = float(np.clip(rng.normal(0.0, self.heading_noise_std),
                              -self.max_turn_rate, self.max_turn_rate))
        new_direction = _direction_from_az_el(az + d_az, el + d_el)

        max_dspeed = self.max_accel * dt
        d_speed = float(np.clip(rng.normal(0.0, self.speed_noise_std), -max_dspeed, max_dspeed))
        new_speed = max(speed + d_speed, 0.0)

        new_vel = new_direction * new_speed
        return _integrate_state(state, new_vel, dt)


@dataclass
class CoordinatedTurn:
    """Широкий вираж постоянной угловой скоростью (большой радиус, не рывок).

    Поворот -- в горизонтальной плоскости XZ (курс), вертикальная (Y)
    составляющая скорости не меняется.
    """

    turn_rate: float = 0.015   # рад/такт (малый -> широкий радиус R=v/turn_rate)

    def propagate(self, state: TargetState, dt: float, rng: np.random.Generator) -> TargetState:
        vx, vy, vz = state.vel
        theta = self.turn_rate * dt
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        new_vx = vx * cos_t - vz * sin_t
        new_vz = vx * sin_t + vz * cos_t
        new_vel = np.array([new_vx, vy, new_vz])
        return _integrate_state(state, new_vel, dt)


@dataclass
class ConstantAccel:
    """Разгон/торможение вдоль текущего вектора скорости (постоянное продольное ускорение)."""

    accel_along_track: float = 1.0   # м/с^2, знак: + разгон, - торможение
    max_accel: float = 3.0

    def propagate(self, state: TargetState, dt: float, rng: np.random.Generator) -> TargetState:
        direction, _ = _direction_and_speed(state.vel)
        a_mag = float(np.clip(self.accel_along_track, -self.max_accel, self.max_accel))
        acc = direction * a_mag
        new_vel = state.vel + acc * dt
        new_pos = state.pos + state.vel * dt + 0.5 * acc * dt * dt
        return state.evolved(pos=new_pos, vel=new_vel, acc=acc)


@dataclass
class WeavingManeuver:
    """Активное уклонение: змейка влево/вправо + горка вверх/вниз + разгон/торможение.

    Детерминированный манёвр (Strategy, stateless -- фаза берётся из `state.tact`):
    курс по азимуту и углу места колеблется синусоидами, скорость -- своей синусоидой.
    Все приращения клипуются по аэро-лимитам (`max_turn_rate`, `max_accel`) -- плавно,
    без рывков. Амплитуды/периоды подобраны так, чтобы манёвр был хорошо виден.
    """

    az_amp: float = 0.55        # рад -- размах змейки по азимуту (лево/право)
    az_period: float = 22.0     # тактов на период лево-право
    el_amp: float = 0.28        # рад -- размах горки (вверх/вниз)
    el_period: float = 15.0     # тактов на период вверх-вниз
    speed_amp: float = 35.0     # м/с -- размах изменения скорости
    speed_period: float = 28.0  # тактов на период разгон-торможение
    max_turn_rate: float = 0.22  # клип |dAz|,|dEl| за такт (рад)
    max_accel: float = 12.0      # клип |dскорость/dt| (м/с^2)
    min_speed: float = 20.0      # нижний предел скорости (не глохнем)

    def propagate(self, state: TargetState, dt: float, rng: np.random.Generator) -> TargetState:
        t = float(state.tact)
        direction, speed = _direction_and_speed(state.vel)
        az, el = _az_el_from_direction(direction)

        w_az = 2.0 * np.pi / self.az_period
        w_el = 2.0 * np.pi / self.el_period
        w_sp = 2.0 * np.pi / self.speed_period

        # приращения = производные синусоид (за такт) -> курс/скорость колеблются с заданной амплитудой
        d_az = float(np.clip(self.az_amp * w_az * np.cos(w_az * t) * dt,
                             -self.max_turn_rate, self.max_turn_rate))
        d_el = float(np.clip(self.el_amp * w_el * np.cos(w_el * t) * dt,
                             -self.max_turn_rate, self.max_turn_rate))
        max_dspeed = self.max_accel * dt
        d_speed = float(np.clip(self.speed_amp * w_sp * np.cos(w_sp * t) * dt,
                                -max_dspeed, max_dspeed))

        new_dir = _direction_from_az_el(az + d_az, el + d_el)
        new_speed = max(speed + d_speed, self.min_speed)
        new_vel = new_dir * new_speed
        return _integrate_state(state, new_vel, dt)
