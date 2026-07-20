"""CA-CFAR детектор по оси дальности (§7 Doc/anti_barrage_math.md).

Алгоритм:
    P̂ = (1/N) Σ Pᵢ    (N = 2·n_train опорных ячеек, P = |C|²)
    T  = α · P̂
    α  = N(P_fa^(-1/N) − 1)   — точная формула для распределения Рэлея

    CUT > T  →  «цель»

Структура окна (симметрично вокруг CUT):
    ← n_train →← n_guard →[ CUT ]← n_guard →← n_train →

Обработка краёв: **усечение** (truncation).
    На краях массива берутся все доступные опорные ячейки (n_eff ≤ N).
    α пересчитывается под n_eff — P_fa остаётся постоянной.
    Если n_eff == 0 (CUT полностью окружён guard-зоной у края), CUT пропускается.

Numpy-эталон — без torch, детерминировано.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..result import SpectralCube

# ── Value Object — одна обнаруженная цель ────────────────────────────────────

@dataclass(frozen=True)
class Detection:
    """Value Object — одна обнаруженная цель.

    Attributes
    ----------
    kx_idx       : индекс угловой ячейки по kx
    ky_idx       : индекс угловой ячейки по ky
    range_bin    : бин дальности (индекс CUT в оси range)
    level_db     : мощность CUT, 10·log10(P_cut / P_global_max), дБ
    threshold_db : порог T,      10·log10(T    / P_global_max), дБ
    kx           : значение kx из оси куба (угловые единицы)
    ky           : значение ky из оси куба (угловые единицы)
    """

    kx_idx: int
    ky_idx: int
    range_bin: int
    level_db: float
    threshold_db: float
    kx: float
    ky: float


# ── Детектор ─────────────────────────────────────────────────────────────────

class CaCfarDetector:
    """Cell-Averaging CFAR детектор по оси дальности.

    Скользящее окно [train | guard | CUT | guard | train] вдоль range-оси.
    Усечение на краях: используются доступные опорные ячейки, α пересчитывается.
    detect() / detect_cell() не мутируют SpectralCube.

    Parameters
    ----------
    pfa       : вероятность ложной тревоги (0 < pfa < 1)
    n_train   : число опорных ячеек с каждой стороны (>= 1)
    n_guard   : число защитных ячеек с каждой стороны (>= 0). Должно перекрывать
                мейнлоуб цели по дальности, иначе цель попадёт в опорные ячейки и
                self-задавит порог (self-masking). Для нашего куба (n_real=16 →
                грубое разрешение, мейнлоуб ~10 бинов) дефолт 4.
    local_max : требовать, чтобы CUT был локальным максимумом по дальности в окне
                ±n_guard. Схлопывает мейнлоуб/юбку цели в один отсчёт (убирает
                дубли-детекции внутри одной угловой ячейки). Угловая кластеризация
                (мейнлоуб по kx/ky) — вне области phase1.
    """

    def __init__(
        self,
        pfa: float = 1e-3,
        n_train: int = 8,
        n_guard: int = 4,
        local_max: bool = True,
    ) -> None:
        if not (0.0 < pfa < 1.0):
            raise ValueError(f"pfa должно быть в (0, 1), получено {pfa}")
        if n_train < 1:
            raise ValueError(f"n_train должно быть >= 1, получено {n_train}")
        if n_guard < 0:
            raise ValueError(f"n_guard должно быть >= 0, получено {n_guard}")
        self._pfa = pfa
        self._n_train = n_train
        self._n_guard = n_guard
        self._local_max = local_max
        n = 2 * n_train
        self._alpha: float = float(n * (pfa ** (-1.0 / n) - 1.0))

    # ── свойства ─────────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        """Пороговый коэффициент α = N(P_fa^(-1/N) − 1), N = 2·n_train."""
        return self._alpha

    @property
    def pfa(self) -> float:
        return self._pfa

    @property
    def n_train(self) -> int:
        return self._n_train

    @property
    def n_guard(self) -> int:
        return self._n_guard

    # ── внутренняя логика ────────────────────────────────────────────────────

    def _cfar_1d(
        self,
        power: np.ndarray,          # (n_range,) float64, P = |C|²
        kx_idx: int,
        ky_idx: int,
        kx_val: float,
        ky_val: float,
        global_max_power: float,
    ) -> list[Detection]:
        """CA-CFAR в одном угловом луче.  Усечение опорной выборки у краёв.

        Для каждого CUT (range_bin) вычисляет опорные ячейки,
        формирует порог и принимает решение «цель / шум».
        """
        n_r = len(power)
        results: list[Detection] = []
        half = self._n_guard + self._n_train    # полуширина всего окна

        for cut in range(n_r):
            # ── границы опорных групп (за пределами guard-зоны) ──────────────
            # Левая группа: power[lft_start : lft_end]
            lft_start = cut - half
            lft_end   = cut - self._n_guard     # не включается в срез

            # Правая группа: power[rgt_start : rgt_end]
            rgt_start = cut + self._n_guard + 1
            rgt_end   = cut + half + 1          # не включается в срез

            # ── ограничение по краям массива (усечение) ───────────────────────
            # ВАЖНО: max(lft_end, 0) — иначе при lft_end<0 Python-срез дал бы
            # power[0:-k], что соответствует хвосту массива.
            lft_cells = power[max(lft_start, 0) : max(lft_end, 0)]
            rgt_cells = power[min(rgt_start, n_r) : min(rgt_end, n_r)]

            n_eff = len(lft_cells) + len(rgt_cells)
            if n_eff == 0:
                # Нет опорных ячеек — CUT пропускается.
                # Это случается только у самого края при n_guard >= n_range//2.
                continue

            p_hat = float((lft_cells.sum() + rgt_cells.sum()) / n_eff)

            # Пересчёт α под фактическое n_eff → P_fa остаётся постоянной
            alpha_eff = float(n_eff * (self._pfa ** (-1.0 / n_eff) - 1.0))
            threshold = alpha_eff * p_hat

            p_cut = float(power[cut])
            if p_cut > threshold:
                # Локальный максимум по дальности (±n_guard): схлопывает мейнлоуб
                # в один отсчёт, убирает дубли внутри угловой ячейки.
                if self._local_max:
                    lo = max(cut - self._n_guard, 0)
                    hi = min(cut + self._n_guard + 1, n_r)
                    if p_cut < float(power[lo:hi].max()):
                        continue
                eps = 1e-30
                gmax = max(global_max_power, eps)
                results.append(Detection(
                    kx_idx=kx_idx,
                    ky_idx=ky_idx,
                    range_bin=cut,
                    level_db=10.0 * float(np.log10(p_cut / gmax)),
                    threshold_db=10.0 * float(np.log10(threshold / gmax)),
                    kx=kx_val,
                    ky=ky_val,
                ))
        return results

    # ── публичный API ─────────────────────────────────────────────────────────

    def detect_cell(
        self,
        cube: SpectralCube,
        ix: int,
        iy: int,
    ) -> list[Detection]:
        """CA-CFAR в одной угловой ячейке (ix, iy).  Куб не мутируется.

        Удобно для тестов и диагностики отдельного луча.

        Parameters
        ----------
        cube : SpectralCube
            Спектральный куб (только чтение).
        ix   : индекс угловой ячейки по kx
        iy   : индекс угловой ячейки по ky

        Returns
        -------
        list[Detection]
        """
        power = cube.magnitude[ix, iy, :].astype(np.float64) ** 2  # P = |C|²
        global_max = float(np.max(cube.magnitude.astype(np.float64) ** 2))
        kx_val = float(cube.kx.values[ix])
        ky_val = float(cube.ky.values[iy])
        return self._cfar_1d(power, ix, iy, kx_val, ky_val, global_max)

    def detect(self, cube: SpectralCube) -> list[Detection]:
        """CA-CFAR по всем угловым ячейкам куба.

        Скользящее окно вдоль оси дальности (axis=2) для каждой ячейки (ix, iy).
        Куб не мутируется.

        Returns
        -------
        list[Detection]
            Список обнаруженных детекций (пустой если ничего не найдено).
        """
        mag = cube.magnitude                            # (nx, ny, n_range) — только чтение
        nx, ny, _ = mag.shape
        power_cube = mag.astype(np.float64) ** 2       # P = |C|², новый массив
        global_max = float(power_cube.max())

        detections: list[Detection] = []
        for ix in range(nx):
            kx_val = float(cube.kx.values[ix])
            for iy in range(ny):
                ky_val = float(cube.ky.values[iy])
                power = power_cube[ix, iy, :]
                detections.extend(
                    self._cfar_1d(power, ix, iy, kx_val, ky_val, global_max)
                )
        return detections
