"""Robust MVDR (Capon) beamformer — адаптивный луч на цель с diagonal loading.

В отличие от `SubspaceNuller` (ортогональная/косая проекция на собств. ВЕКТОРЫ
подпространства помехи, где diagonal loading — no-op, см. docstring
`SubspaceNuller.loading`), MVDR строит веса через ОБРАЩЕНИЕ ковариации `R⁻¹`.
Diagonal loading здесь меняет сам оператор обращения — при малой выборке
снапшотов K (K < M, M = nx*ny) выборочная R вырождена/плохо обусловлена,
`R⁻¹` численно нестабильна (огромные веса на шумовых направлениях). Loading
регуляризует R перед обращением → это "robust Capon" (стандартный приём
адаптивного формирования луча).

Numpy-эталон (np.linalg.solve, R эрмитова).  GPU / torch — вне рамок задачи.
"""
from __future__ import annotations

import numpy as np

from ._covariance import apply_diagonal_loading, reshape_datacube, sample_covariance


class RobustMvdrNuller:
    """MVDR (Capon) beamformer: адаптивный луч на цель, помеха подавляется адаптивно.

    Формула весов: w = (R⁻¹ a) / (aᴴ R⁻¹ a), где a — вектор наведения на цель
    (target_steering), R — выборочная ковариация [+ diagonal loading].
    Выход: y = wᴴ X (K,) — очищенный сигнал вдоль направления цели.

    Параметры
    ----------
    target_steering : np.ndarray
        Вектор наведения на цель, форма (M,) или (nx, ny) → будет выпрямлен (ravel).
        M = nx*ny (число элементов решётки).
    loading : float
        Коэффициент **diagonal loading**, >= 0. Дефолт 0.01 (loading > 0
        обязателен по умолчанию: без него R⁻¹ вырождена/шумит при K < M).
        R' = R + loading · (tr(R)/M) · I — масштаб tr(R)/M — средняя мощность
        на канал, чтобы loading был безразмерным (не зависел от абс. уровня
        сигнала). Чем меньше K относительно M, тем важнее loading > 0.
    """

    def __init__(self, target_steering: np.ndarray, loading: float = 0.01) -> None:
        if loading < 0.0:
            raise ValueError("loading должно быть >= 0")
        self._target_steering = np.asarray(target_steering).astype(np.complex128).ravel()
        self._loading = loading

    # ── внутренние вспомогательные ──────────────────────────────────────────

    def _cov(self, datacube: np.ndarray) -> np.ndarray:
        """Ковариация с diagonal loading: R = X@X.H/K + loading·(tr(R)/M)·I.

        Возвращает R (M, M), эрмитова, complex128.
        """
        x_mat = reshape_datacube(datacube)
        r_cov = sample_covariance(x_mat)
        return apply_diagonal_loading(r_cov, self._loading)

    # ── публичный API ───────────────────────────────────────────────────────

    def weights(self, datacube: np.ndarray) -> np.ndarray:
        """Веса MVDR-луча w = R⁻¹a / (aᴴ R⁻¹ a).  Вход не мутируется.

        Обращение через `np.linalg.solve` (устойчивее явного `inv`).

        Returns
        -------
        np.ndarray
            w, форма (M,), complex128.
        """
        r_cov = self._cov(datacube)
        a_t = self._target_steering
        r_inv_a = np.linalg.solve(r_cov, a_t)               # R⁻¹ a  (M,)
        denom = a_t.conj() @ r_inv_a                         # aᴴ R⁻¹ a (скаляр)
        return r_inv_a / denom

    def apply(self, datacube: np.ndarray) -> np.ndarray:
        """Применить MVDR-луч: y = wᴴ X.  Вход не мутируется.

        Parameters
        ----------
        datacube : np.ndarray
            Сырой куб (nx, ny, K) complex.

        Returns
        -------
        np.ndarray
            Луч на цель, форма (K,), complex128.
        """
        x_mat = reshape_datacube(datacube)                    # X (M, K)
        w = self.weights(datacube)                            # (M,)
        return w.conj() @ x_mat                               # yᴴ = wᴴ X → (K,)
