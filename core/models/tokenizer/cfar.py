"""OsCfarDetector -- OS-CFAR порог выброса + поиск до 5 пиков (гл.4 §4.6, гл.4-бис §4-бис.3).

Обобщённый (2D/3D) детектор пиков над картой/объёмом `P = |A|²`:
    главный лепесток   -- блок полуширины `main_half`  (3 в 2D / 3x3x3 в 3D при half=1)
    охранная зона      -- блок полуширины `guard_half` (5 в 2D / 5x5x5 в 3D при half=2)
    обучающая выборка   -- ячейки внутри guard-блока, но вне main-блока (кольцо)

Идея реюзана из `core/models/anti_barrage/cfar.py` (CaCfarDetector: скользящее окно
train/guard вокруг CUT, α по Рэлеевской формуле, усечение на краях) и
`core/snr/estimator.py` (CFAR ref-window вокруг пика). Отличие OS- от CA-CFAR:
шумовая оценка -- ПРОЦЕНТИЛЬ (order statistic) обучающих ячеек, а не среднее --
устойчивее к нескольким мешающим пикам в референс-окне (§4-бис.3: "OS-CFAR --
объёмные обучающие/охранные ячейки").

⚠️ Честная девиация (для ревью Кодо): α считается по приближённой Рэлеевской
CA-CFAR формуле `α = N·(P_fa^(-1/N) − 1)` (как в `CaCfarDetector`), применённой
к N обучающим ячейкам -- НЕ точная OS-CFAR Pfa-формула (та требует Beta-функции
от ранга k, зависимость от N и k заметно отличается от CA). Для 16x16(xL)-карты
и разреженного (<=5) поиска пиков это достаточно; точную калибровку имеет смысл
сделать отдельно, если понадобится жёсткая Pfa-гарантия.
"""
from __future__ import annotations

import numpy as np


class OsCfarDetector:
    """OS-CFAR (Strategy): единое ядро для 2D-среза (L=1) и 3D-объёма (L>1).

    Parameters
    ----------
    pfa        : вероятность ложной тревоги (0 < pfa < 1).
    main_half  : полуширина главного лепестка (охранный блок вокруг CUT, 1 -> 3x3(x3)).
    guard_half : полуширина обучающей зоны (5x5(x5) при half=2, гл.4-бис §4-бис.3).
    percentile : процентиль обучающих ячеек как шумовая оценка (75 -> устойчив к
                 ~четверти обучающих ячеек, "засорённых" соседним пиком).
    max_peaks  : максимум пиков в токене (гл.4 §4.6 -- до 5; больше -> само "размазано").
    """

    def __init__(
        self,
        pfa: float = 1e-3,
        main_half: int = 1,
        guard_half: int = 2,
        percentile: float = 75.0,
        max_peaks: int = 5,
    ) -> None:
        if not (0.0 < pfa < 1.0):
            raise ValueError(f"pfa должно быть в (0, 1), получено {pfa}")
        if guard_half < main_half:
            raise ValueError(f"guard_half ({guard_half}) должен быть >= main_half ({main_half})")
        if not (0.0 <= percentile <= 100.0):
            raise ValueError(f"percentile должен быть в [0,100], получено {percentile}")
        if max_peaks < 1:
            raise ValueError(f"max_peaks должен быть >= 1, получено {max_peaks}")
        self._pfa = pfa
        self._main_half = main_half
        self._guard_half = guard_half
        self._percentile = percentile
        self._max_peaks = max_peaks

    @property
    def max_peaks(self) -> int:
        return self._max_peaks

    # ── внутреннее ───────────────────────────────────────────────────────────

    @staticmethod
    def _block_bounds(shape: tuple[int, ...], center: tuple[int, ...],
                       half: int) -> tuple[slice, ...]:
        return tuple(
            slice(max(0, c - half), min(n, c + half + 1))
            for c, n in zip(center, shape, strict=True)
        )

    def _train_cells(self, p: np.ndarray, center: tuple[int, ...]) -> np.ndarray:
        """Ячейки обучающей зоны: внутри guard-блока, вне main-блока (кольцо)."""
        guard_mask = np.zeros(p.shape, dtype=bool)
        guard_mask[self._block_bounds(p.shape, center, self._guard_half)] = True
        guard_mask[self._block_bounds(p.shape, center, self._main_half)] = False
        return p[guard_mask]

    def _alpha_for(self, n_eff: int) -> float:
        if n_eff <= 0:
            return float("inf")
        return float(n_eff * (self._pfa ** (-1.0 / n_eff) - 1.0))

    def cell_threshold(self, p: np.ndarray, center: tuple[int, ...]) -> float:
        """Порог CFAR в одной ячейке `center` (усечение train-кольца на краях)."""
        train = self._train_cells(p, center)
        n_eff = train.size
        if n_eff == 0:
            return float("inf")
        noise_est = float(np.percentile(train, self._percentile))
        alpha = self._alpha_for(n_eff)
        return alpha * max(noise_est, 1e-30)

    def detect_mask(self, power: np.ndarray) -> np.ndarray:
        """Булева маска -- CUT > порог(CUT), по каждой ячейке карты/объёма.

        Не мутирует `power`. Реализовано python-циклом по ячейкам (карта мала --
        16x16(xL), <=256·L ячеек; корректность важнее скорости на этом этапе,
        см. приоритеты проекта -- сначала работоспособность/корректность).
        """
        p = np.asarray(power, dtype=np.float64)
        mask = np.zeros(p.shape, dtype=bool)
        for idx in np.ndindex(p.shape):
            thr = self.cell_threshold(p, idx)
            if p[idx] > thr:
                mask[idx] = True
        return mask

    def find_peaks(
        self, power: np.ndarray, max_peaks: int | None = None,
    ) -> list[tuple[tuple[int, ...], float]]:
        """До `max_peaks` локальных максимумов над CFAR-порогом.

        Non-max suppression: после каждого найденного пика зануляем guard-блок
        вокруг него в рабочей копии, чтобы не задвоить один и тот же лепесток.
        Возвращает список `(индекс, амплитуда)` в порядке убывания амплитуды.
        """
        n_max = self._max_peaks if max_peaks is None else max_peaks
        p = np.asarray(power, dtype=np.float64)
        mask = self.detect_mask(p)
        working = p.copy()
        working[~mask] = -np.inf

        peaks: list[tuple[tuple[int, ...], float]] = []
        for _ in range(n_max):
            flat = int(np.argmax(working))
            idx = np.unravel_index(flat, working.shape)
            val = float(working[idx])
            if not np.isfinite(val):
                break
            peaks.append((idx, float(p[idx])))
            sl = self._block_bounds(working.shape, idx, self._guard_half)
            working[sl] = -np.inf
        return peaks
