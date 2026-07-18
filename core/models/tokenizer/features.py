"""FeatureExtractor -- 6 инженерных признаков угловой/объёмной карты (гл.4 §4.5).

Pure Fabrication: чистая редукция энергетической карты `P = |A|²` в короткий
вектор признаков. Работает как на плоском срезе (2D, `L=1`), так и на объёме
(3D, `L>1`, гл.4-бис §4-бис.3) -- один и тот же код, размерность `power.ndim`
определяет, "лепесток" 3x3 это или 3x3x3 (унификация ядра, не спец-случаи).

ВАЖНО (E7 таска): вход -- уже энергетическая карта `P = |A|²`. `SpectralCube.magnitude`
хранит `|A|` (амплитуду), НЕ `|A|²` -- возведение в квадрат делает вызывающий код
(`VolumeTokenizer`), сюда попадает готовое `P`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureVector:
    """6 признаков среза/объёма (гл.4 §4.5) -- Value Object.

    ВАЖНО (нормировка на M, F9): после перехода на апертуру i×j (nx != ny,
    паддинг до 2ⁿ независимо по осям, `ArrayConfig.padded_shape()`) число ячеек
    угловой карты `M = p.size` больше НЕ фиксировано (было 256 = 16x16).
    `pr`, `energy`, `max_mean` растут пропорционально M (участие/энергия/контраст
    считаются по абсолютным суммам) -- без нормировки якоря триажа §4.11
    (калиброванные на M=256) плыли бы при другой апертуре. Поэтому здесь они
    нормированы делением на `m = float(p.size)`, что делает их ИНВАРИАНТНЫМИ
    к размеру апертуры (проверено `tests/test_tokenizer.py`,
    `test_pr_invariant_to_aperture_size`):
      - `pr`       -- PR/M, "participation fraction" ∈ (0, 1]: цель ~0.014,
                      заградка ~0.074, шум ~0.5.
      - `energy`   -- S1/M, средняя мощность на ячейку.
      - `max_mean` -- (max/mean)/M, контраст пика, нормированный на размер карты.
    `hoyer`, `main_frac`, `lobe_ratio` инвариантны к M УЖЕ по построению (это
    отношения однородных величин, размер карты в них сокращается) -- НЕ трогаем.
    """

    pr: float          # Participation Ratio, нормированный: (S1²/S2) / M ∈ (0,1]
    hoyer: float        # индекс Хойера, [0,1] -- инвариантен к M по построению
    main_frac: float    # доля энергии в главном лепестке -- инвариантен к M
    lobe_ratio: float   # 2-й лепесток / главный -- инвариантен к M
    max_mean: float     # контраст пика, нормированный: (max/mean) / M
    energy: float       # средняя мощность на ячейку: S1 / M


class FeatureExtractor:
    """Извлекает `FeatureVector` из энергетической карты `P` (гл.4 §4.5, гл.4-бис §4-бис.3).

    Parameters
    ----------
    main_half  : полуширина главного лепестка (1 -> блок 3 по каждой оси, т.е. 3x3 в 2D
                 или 3x3x3 в 3D).
    guard_half : полуширина охранной зоны вокруг главного пика перед поиском 2-го лепестка
                 (2 -> блок 5 по каждой оси, т.е. 5x5 в 2D или 5x5x5 в 3D).
    """

    _EPS = 1e-30

    def __init__(self, main_half: int = 1, guard_half: int = 2) -> None:
        if main_half < 0:
            raise ValueError(f"main_half должен быть >= 0, получено {main_half}")
        if guard_half < main_half:
            raise ValueError(
                f"guard_half ({guard_half}) должен быть >= main_half ({main_half})"
            )
        self._main_half = main_half
        self._guard_half = guard_half

    def extract(self, power: np.ndarray) -> FeatureVector:
        """`power` -- P = |A|², произвольной размерности (2D срез или 3D объём).

        `pr`/`energy`/`max_mean` нормируются на `m = p.size` (число ячеек карты)
        -- инвариантность к размеру апертуры i×j после паддинга (F9, см.
        докстринг `FeatureVector`).
        """
        p = np.asarray(power, dtype=np.float64)
        m = float(p.size)
        s1 = float(p.sum())
        s2 = float((p * p).sum())

        pr = ((s1 * s1) / max(s2, self._EPS)) / m

        sqrt_m = np.sqrt(m)
        hoyer = (float((sqrt_m - s1 / np.sqrt(max(s2, self._EPS))) / (sqrt_m - 1.0))
                 if m > 1.0 else 0.0)

        peak_idx = np.unravel_index(int(np.argmax(p)), p.shape)
        main_sum = self._block_sum(p, peak_idx, self._main_half)
        main_frac = main_sum / max(s1, self._EPS)

        masked = p.copy()
        self._zero_block(masked, peak_idx, self._guard_half)
        second_idx = np.unravel_index(int(np.argmax(masked)), masked.shape)
        second_sum = self._block_sum(masked, second_idx, self._main_half)
        lobe_ratio = second_sum / max(main_sum, self._EPS)

        max_mean = (float(p.max()) / max(float(p.mean()), self._EPS)) / m
        energy = s1 / m

        return FeatureVector(
            pr=pr, hoyer=hoyer, main_frac=main_frac,
            lobe_ratio=lobe_ratio, max_mean=max_mean, energy=energy,
        )

    # ── внутренние редукции (блок вокруг индекса, с усечением на краях) ────────

    @staticmethod
    def _block_slices(shape: tuple[int, ...], center: tuple[int, ...],
                       half: int) -> tuple[slice, ...]:
        return tuple(
            slice(max(0, c - half), min(n, c + half + 1))
            for c, n in zip(center, shape, strict=True)
        )

    def _block_sum(self, p: np.ndarray, center: tuple[int, ...], half: int) -> float:
        sl = self._block_slices(p.shape, center, half)
        return float(p[sl].sum())

    def _zero_block(self, p: np.ndarray, center: tuple[int, ...], half: int) -> None:
        sl = self._block_slices(p.shape, center, half)
        p[sl] = 0.0
