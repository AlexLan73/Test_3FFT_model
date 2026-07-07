"""Пространственный нуллер подпространства помехи (SubspaceNuller).

Реализует §4.1 (ортогональная проекция) и §4.2 (косая проекция) из
Doc/anti_barrage_math.md.  Работает в домене элементов решётки ДО Fft3DModel.

Numpy-эталон (np.linalg.eigh, R эрмитова).  GPU / torch — в phase2.

Соглашение о переменных (в docstring/комментариях — мат. обозначения):
  x_mat  ← X  (данные, M×K)   r_cov  ← R  (ковариация)
  e_vecs ← E  (собств. векторы) e_jam  ← E_J (подпростр. помехи)
  p_perp ← P⊥ (ортопроектор)   k_snap ← K  (число отсчётов)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Value Object — отчёт нуллера
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NullerReport:
    """Аналитический отчёт о помеховой обстановке.

    Attributes
    ----------
    lambda_ratio : float
        Отношение λ₁/σ_n² (доминантное с.з. / уровень шума).
    occupancy : float
        Доля дальностных бинов (после FFT) с выраженной энергией, ∈ [0, 1].
    is_barrage : bool
        True если lambda_ratio > _LAMBDA_ETA и occupancy > _OCC_ETA
        (оба критерия §3 выполнены).
    """
    lambda_ratio: float
    occupancy: float
    is_barrage: bool


# ──────────────────────────────────────────────────────────────────────────────
# Основной класс
# ──────────────────────────────────────────────────────────────────────────────

class SubspaceNuller:
    """Угловое подавление заградительной помехи через EVD ковариационной матрицы.

    Параметры
    ----------
    n_jammers : int
        Число подавляемых источников (доминантных собств. векторов).
    oblique : bool
        False → ортогональная проекция (§4.1, быстро).
        True  → косая проекция (§4.2), сохраняет амплитуду цели точно.
        Требует target_steering.
    target_steering : np.ndarray | None
        Вектор наведения на цель формы (M,), M = nx*ny.
        Используется только при oblique=True.
    """

    # Пороги критерия «это barrage»
    _LAMBDA_ETA: float = 5.0    # минимальное λ₁/σ_n²
    _OCC_ETA: float = 0.40      # минимальная occupancy по дальности (после FFT)

    def __init__(
        self,
        n_jammers: int = 1,
        oblique: bool = False,
        target_steering: np.ndarray | None = None,
    ) -> None:
        if n_jammers < 1:
            raise ValueError("n_jammers должно быть >= 1")
        self._n_jammers = n_jammers
        self._oblique = oblique
        self._target_steering = target_steering

    # ── внутренние вспомогательные ──────────────────────────────────────────

    def _reshape(self, datacube: np.ndarray) -> np.ndarray:
        """datacube (nx, ny, K) → x_mat (M, K), M = nx*ny.  complex128 для точности."""
        nx, ny, k_snap = datacube.shape
        return datacube.reshape(nx * ny, k_snap).astype(np.complex128, copy=False)

    def _eigh(self, datacube: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ковариация (R = X@X.H/K) + EVD.

        Возвращает (x_mat[M,K], lambdas[M] по возрастанию, e_jam[M,n_jammers]).
        e_jam — топ-n_jammers собств. векторов (подпространство помехи).
        """
        x_mat = self._reshape(datacube)                     # X (M, K)
        k_snap = x_mat.shape[1]
        r_cov = (x_mat @ x_mat.conj().T) / k_snap          # R (M, M), эрмитова
        lambdas, e_vecs = np.linalg.eigh(r_cov)            # ascending: λ₀ ≤ … ≤ λ_{M-1}
        e_jam = e_vecs[:, -self._n_jammers:]                # доминантные (помеха)
        return x_mat, lambdas, e_jam

    @staticmethod
    def _ortho_projector(e_jam: np.ndarray) -> np.ndarray:
        """P⊥ = I − E_J E_Jᴴ.  Ортогональный проектор на шумовое подпространство."""
        m_elem = e_jam.shape[0]
        return np.eye(m_elem, dtype=np.complex128) - e_jam @ e_jam.conj().T

    # ── публичный API ───────────────────────────────────────────────────────

    def decompose(
        self, datacube: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть (p_perp[M,M], e_jam[M,n_jammers]) для диагностики/тестов.

        p_perp ← P⊥ (ортопроектор),  e_jam ← E_J (базис помехи).
        """
        _, _, e_jam = self._eigh(datacube)
        return self._ortho_projector(e_jam), e_jam

    def apply(self, datacube: np.ndarray) -> np.ndarray:
        """Подавить помеху.  Вход НЕ мутируется.

        Parameters
        ----------
        datacube : np.ndarray
            Сырой куб (nx, ny, K) complex.

        Returns
        -------
        np.ndarray
            Очищенный куб (nx, ny, K), dtype = dtype входа.
        """
        nx, ny, k_snap = datacube.shape
        dtype_in = datacube.dtype
        x_mat, _lambdas, e_jam = self._eigh(datacube)      # X (M,K), E_J (M,J)
        p_perp = self._ortho_projector(e_jam)               # P⊥ (M,M)

        if self._oblique and self._target_steering is not None:
            # §4.2: E_{t|J} = a_t (a_tᴴ P⊥ a_t)⁻¹ a_tᴴ P⊥
            # Свойства: E_{t|J} a_t = a_t, E_{t|J} a_J = 0
            a_t = self._target_steering.astype(np.complex128).ravel()  # (M,)
            p_perp_at = p_perp @ a_t                                    # P⊥ a_t (M,)
            denom = float(np.real(a_t.conj() @ p_perp_at))
            if abs(denom) > 1e-10:
                # Y = E_{t|J} X = a_t * ((P⊥ a_t)ᴴ X) / denom
                coeffs = p_perp_at.conj() @ x_mat           # (K,) проекции снимков
                y_out = np.outer(a_t, coeffs) / denom        # (M, K)
            else:
                # Вырожденный случай (цель совпала с помехой) — ортогональная проекция
                y_out = p_perp @ x_mat
        else:
            # §4.1: ортогональная проекция  Y = P⊥ X
            y_out = p_perp @ x_mat                          # (M, K)

        return y_out.reshape(nx, ny, k_snap).astype(dtype_in)

    def report(self, datacube: np.ndarray) -> NullerReport:
        """Оценить наличие barrage.  Вход не мутируется.

        λ₁/σ_n² — пространственный критерий (§3).
        occupancy — оценивается после FFT по дальностной оси (§3).

        Returns
        -------
        NullerReport
        """
        _nx, _ny, k_snap = datacube.shape
        x_mat, lambdas, _e_jam = self._eigh(datacube)

        # σ_n² ≈ медиана «шумовых» собственных значений (без топ-n_jammers).
        # При K < M матрица R имеет ранг ≤ K; нулевые с.з. — индексы [0..M-K-1].
        # Берём только ненулевые (последние K), исключая верхние n_jammers.
        n_total = len(lambdas)                              # = M = nx*ny
        k_dim   = x_mat.shape[1]                            # = k_snap (ранг ≤ K)
        noise_start = n_total - k_dim
        noise_end   = n_total - self._n_jammers
        if noise_end > noise_start:
            sigma_n2 = float(np.median(lambdas[noise_start:noise_end]))
        else:
            sigma_n2 = float(lambdas[noise_start])
        lambda1 = float(lambdas[-1])
        lambda_ratio = lambda1 / max(sigma_n2, 1e-30)

        # Occupancy: доля дальностных бинов с выраженной энергией после FFT.
        # Barrage — белый шум → равномерный спектр; цель — тон → один пик.
        x_fft = np.fft.fft(x_mat, axis=1)                  # (M, K) спектр по дальности
        range_energy = np.mean(np.abs(x_fft) ** 2, axis=0)  # (K,) средняя мощность
        occ_thr = range_energy.mean() * 0.5
        occupancy = float((range_energy > occ_thr).sum()) / k_snap

        is_barrage = (lambda_ratio > self._LAMBDA_ETA) and (occupancy > self._OCC_ETA)
        return NullerReport(
            lambda_ratio=lambda_ratio,
            occupancy=occupancy,
            is_barrage=is_barrage,
        )
