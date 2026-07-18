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

Множитель α считается по ТОЧНОЙ OS-CFAR Pfa-формуле (Rohling 1983, экспоненциальная
мощность = Рэлеевская амплитуда): порог `T = α·X_(k)`, где `X_(k)` -- k-я порядковая
статистика (по возрастанию) из N обучающих ячеек,

    Pfa(α, N, k) = ∏_{i=0}^{k-1} (N - i) / (N - i + α)

монотонно убывающая по α -- α ищется численно (bisection, `_alpha_os`). Ранее
использовалась приближённая CA-CFAR формула `α = N·(P_fa^(-1/N) − 1)` (оставлена
как приватный fallback `_alpha_for`, не используется в `cell_threshold`).
"""
from __future__ import annotations

import numpy as np

_BISECT_ITERS = 100
_HI_START = 1.0
_HI_MAX = 1e15


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
        self._alpha_cache: dict[tuple[int, int], float] = {}

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
        """Приближённая CA-CFAR формула α = N·(Pfa^(-1/N) − 1) -- НЕ используется
        в `cell_threshold` (см. `_alpha_os`); оставлена приватным fallback/для
        сравнения (smoke: α_OS vs α_CA)."""
        if n_eff <= 0:
            return float("inf")
        return float(n_eff * (self._pfa ** (-1.0 / n_eff) - 1.0))

    @staticmethod
    def _os_pfa(alpha: float, n: int, k: int) -> float:
        """Точная OS-CFAR Pfa (Rohling 1983): ∏_{i=0}^{k-1} (n-i)/(n-i+alpha).

        Экспоненциальная мощность (Рэлеевская амплитуда), T = alpha·X_(k), где
        X_(k) -- k-я порядковая статистика (по возрастанию) из n обучающих ячеек.
        Монотонно убывает по alpha: alpha=0 -> Pfa=1, alpha->inf -> Pfa->0.
        """
        if not (1 <= k <= n):
            raise ValueError(f"k={k} должен быть в [1, n={n}]")
        if alpha < 0.0:
            raise ValueError(f"alpha={alpha} должен быть >= 0")
        pfa = 1.0
        for i in range(k):
            pfa *= (n - i) / (n - i + alpha)
        return pfa

    @classmethod
    def _alpha_os(cls, n: int, k: int, pfa: float) -> float:
        """Множитель α, дающий заданный `pfa` для точной OS-CFAR формулы (bisection).

        `_os_pfa` монотонно убывает по alpha -> единственный корень, ищем bisection'ом:
        alpha=0 даёт Pfa=1 (>= любого целевого pfa<1), растим верхнюю границу, пока
        Pfa(hi) не опустится ниже pfa, затем ~100 итераций половинного деления.
        """
        if not (1 <= k <= n):
            raise ValueError(f"k={k} должен быть в [1, n={n}]")
        lo, hi = 0.0, _HI_START
        while cls._os_pfa(hi, n, k) > pfa and hi < _HI_MAX:
            hi *= 2.0
        for _ in range(_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            if cls._os_pfa(mid, n, k) > pfa:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _rank_for_percentile(self, n_eff: int) -> int:
        """k-й ранг (1..n_eff) обучающих ячеек, соответствующий `self._percentile`."""
        return max(1, min(n_eff, int(round(self._percentile / 100.0 * n_eff))))

    def cell_threshold(self, p: np.ndarray, center: tuple[int, ...]) -> float:
        """Порог CFAR в одной ячейке `center` (усечение train-кольца на краях).

        Шумовая оценка -- k-я порядковая статистика (`np.partition`, O(n), СТРОГАЯ,
        не интерполяция percentile) обучающих ячеек; множитель α -- точная OS-CFAR
        Pfa-формула (`_alpha_os`, Rohling), пересчитывается per-ячейка (n_eff/k
        меняются на краях из-за усечения train-кольца) и кешируется по (n_eff, k).
        """
        train = self._train_cells(p, center)
        n_eff = train.size
        if n_eff == 0:
            return float("inf")
        k = self._rank_for_percentile(n_eff)
        noise_est = float(np.partition(train, k - 1)[k - 1])
        cache_key = (n_eff, k)
        alpha = self._alpha_cache.get(cache_key)
        if alpha is None:
            alpha = self._alpha_os(n_eff, k, self._pfa)
            self._alpha_cache[cache_key] = alpha
        return alpha * max(noise_est, 1e-30)

    def detect_mask(self, power: np.ndarray) -> np.ndarray:
        """Булева маска -- CUT > порог(CUT), по каждой ячейке карты/объёма.

        Не мутирует `power`. Реализовано python-циклом по ячейкам (карта мала --
        i×j(xL), типично <=256·L ячеек; корректность важнее скорости на этом этапе,
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
