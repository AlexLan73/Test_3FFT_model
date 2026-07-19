"""Параболическое уточнение пика куба — суб-биновая точность (TASK_ex1_search_p2, решение Alex).

Целочисленный argmax даёт позицию пика с ошибкой до ±0.5 бина. Истинный источник
почти всегда сидит МЕЖДУ бинами (дробный угол/частота) — вершина лог-параболы,
проведённой через 3 точки вдоль оси, восстанавливает дробную часть:

    δ = ½(P₋ − P₊) / (P₋ − 2·P₀ + P₊),   P = log₁₀(мощность), δ ∈ [−½, ½]

Для гауссова (оконного) пика лог-парабола точна; для FFT-пика с тэйпером
(Хэмминг/Ханн) остаточная ошибка ≪ 0.1 бина. По каждой оси куба (kx, ky, range)
поправка считается независимо — это 1D-парабола из `demo/ex1_am_line/matched.py`
(`parabolic_refine_hz`), поднятая на N-мерный случай.

На краю оси (нет соседа) поправка не считается (δ=0): угловые оси куба
центрированы `fftshift`'ом — заворот через край невалиден.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-30    # пол под log10: нулевая мощность не роняет вычисление


def parabolic_delta(p_minus: float, p_zero: float, p_plus: float) -> float:
    """Смещение вершины лог-параболы от центрального бина, δ ∈ [−½, ½].

    Вход — МОЩНОСТИ (|·|²) трёх соседних бинов. Невыпуклый/вырожденный случай
    (знаменатель ≥ 0 — центральная точка не пик) — честный 0.0, не экстраполяция.
    """
    l_m, l_0, l_p = (float(np.log10(max(v, _EPS))) for v in (p_minus, p_zero, p_plus))
    denom = l_m - 2.0 * l_0 + l_p
    if denom >= 0.0:
        return 0.0
    delta = 0.5 * (l_m - l_p) / denom
    return float(np.clip(delta, -0.5, 0.5))


@dataclass(frozen=True)
class RefinedPeak:
    """Уточнённый пик N-мерного массива (VO): целый argmax + суб-биновые поправки."""

    index: tuple[int, ...]      # целочисленный пик (бины)
    delta: tuple[float, ...]    # поправка по каждой оси, каждая в [−½, ½]

    @property
    def frac_index(self) -> tuple[float, ...]:
        """Дробный индекс пика: index + delta по каждой оси."""
        return tuple(i + d for i, d in zip(self.index, self.delta, strict=True))


def refine_peak(power: np.ndarray, index: tuple[int, ...] | None = None) -> RefinedPeak:
    """Суб-биновое уточнение пика массива мощности (куб 3D — частный случай N-D).

    `index=None` — уточняется глобальный argmax; иначе — заданный пик (например,
    из `OsCfarDetector.find_peaks` или грубой карты). По каждой оси — независимая
    лог-парабола через (i−1, i, i+1). Вход не мутируется.
    """
    p = np.asarray(power, dtype=np.float64)
    if index is None:
        index = tuple(int(v) for v in np.unravel_index(int(np.argmax(p)), p.shape))
    if len(index) != p.ndim:
        raise ValueError(f"index размерности {len(index)} не совпадает с ndim={p.ndim}")

    deltas: list[float] = []
    for ax, i in enumerate(index):
        if i <= 0 or i >= p.shape[ax] - 1:
            deltas.append(0.0)      # край оси: соседа нет, честно без уточнения
            continue
        line = p[index[:ax] + (slice(i - 1, i + 2),) + index[ax + 1:]]
        deltas.append(parabolic_delta(float(line[0]), float(line[1]), float(line[2])))
    return RefinedPeak(index=tuple(index), delta=tuple(deltas))


def axis_value_at(values: np.ndarray, frac_index: float) -> float:
    """Физическое значение оси (`Axis.values`) в дробном индексе (линейная интерполяция)."""
    return float(np.interp(frac_index, np.arange(len(values)), values))
