"""SquareView -- прямоугольник i×j (reduce по range) + argmax + окрестность +-N (Strategy, P5).

Контрольный вид (упрощённый) над общим низом `SpectralCube` (kx,ky,range) -- ОБЩИЙ для
обеих веток заполнения (`LfmToCube`/`AmToCube`, SPEC §2): «reduce по range-оси -> карта
i×j энергии + argmax дальности» -- стадии Q1/Q2/Q3 (SPEC §5: профиль по Z / угловые
плоскости / токены). Апертура необязательно квадратная (`nx != ny` допустимо, F9).
Полный объёмный токенизатор (OS-CFAR 3D + 3D-признаки, гл.4-бис) --
этап детектора, НЕ здесь (SPEC §2: "прототип -- контрольный вид").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..models.result import SpectralCube


@dataclass(frozen=True)
class SquareToken:
    """Ground-truth токен контрольного вида (reduce+argmax) -- НЕ полный OS-CFAR-токен."""

    ix: int
    iy: int
    range_bin: int
    range_m: float
    magnitude: float


@dataclass(frozen=True)
class SquareView:
    """reduce по 3-й (range) оси -> карта i×j (kx,ky) + argmax + блок окрестности +-N.

    `reduce_mode`      -- "max" (пик по дальности, годится для точки/протяжённой цели)
                           или "sum" (энергия по дальности, годится для заграда -- полоса
                           во всех окнах даёт большую сумму, а не один узкий пик).
    `neighbor_planes`  -- +-N плоскостей вокруг argmax-бина (SPEC §5.1, дефолт из
                           `ProjectConfig.viz_neighbor_planes`=5).
    """

    reduce_mode: Literal["max", "sum"] = "max"
    neighbor_planes: int = 5

    def __post_init__(self) -> None:
        if self.neighbor_planes < 0:
            raise ValueError(f"neighbor_planes не может быть отрицательным, получено {self.neighbor_planes}")

    def reduce_square(self, cube: SpectralCube) -> np.ndarray:
        """Карта i×j (kx,ky) -- reduce по range-оси (`reduce_mode`)."""
        mag = cube.magnitude
        if self.reduce_mode == "max":
            return mag.max(axis=2)
        if self.reduce_mode == "sum":
            return mag.sum(axis=2)
        raise ValueError(f"неизвестный reduce_mode={self.reduce_mode!r}")

    def argmax_range(self, cube: SpectralCube) -> tuple[int, int, int]:
        """`(ix, iy, iz)` глобального максимума куба -- уточнение позиции цели (SPEC §2)."""
        mag = cube.magnitude
        flat = int(np.argmax(mag))
        ix, iy, iz = np.unravel_index(flat, mag.shape)
        return int(ix), int(iy), int(iz)

    def neighbor_block(self, cube: SpectralCube, iz: int) -> np.ndarray:
        """Срез `(nx, ny, <=2N+1)` -- окрестность +-`neighbor_planes` бинов вокруг `iz`.

        На границе куба блок короче (обрезка, а не паддинг) -- "2 бина на границе"
        (TASK): критерий приёмки говорит про 1 окно ±N (2 на границе), НЕ про паддинг.
        """
        n = cube.magnitude.shape[2]
        lo = max(0, iz - self.neighbor_planes)
        hi = min(n, iz + self.neighbor_planes + 1)
        return cube.magnitude[:, :, lo:hi]

    def range_profile(self, cube: SpectralCube, ix: int, iy: int) -> np.ndarray:
        """Профиль магнитуды по дальности в ячейке `(ix, iy)` -- для измерения ширины пика."""
        return cube.magnitude[ix, iy, :]

    def tokenize(self, cube: SpectralCube, threshold_db: float = -10.0) -> list[SquareToken]:
        """Упрощённые ground-truth токены (reduce+argmax по квадрату) -- НЕ OS-CFAR (детектор).

        `threshold_db` -- порог относительно максимума СВЁРНУТОГО (reduce) квадрата.
        """
        square = self.reduce_square(cube)
        square_db = 20.0 * np.log10(square + 1e-12)
        square_db = square_db - square_db.max()

        tokens: list[SquareToken] = []
        ix_arr, iy_arr = np.where(square_db >= threshold_db)
        for ix, iy in zip(ix_arr.tolist(), iy_arr.tolist(), strict=True):
            profile = self.range_profile(cube, ix, iy)
            iz = int(np.argmax(profile))
            tokens.append(SquareToken(
                ix=int(ix), iy=int(iy), range_bin=iz,
                range_m=float(cube.range.values[iz]), magnitude=float(profile[iz]),
            ))
        return tokens
