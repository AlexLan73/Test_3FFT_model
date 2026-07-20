"""Общие вычисления ковариации решётки (Pure Fabrication).

Вынесено из `SubspaceNuller._eigh` (nuller.py) и `RobustMvdrNuller._cov` (mvdr.py) —
обе формулы (reshape датакуба в матрицу снимков, выборочная ковариация, diagonal
loading) были продублированы дословно (см. `.claude/rules/07-math-in-core.md`,
находка architecture review 2026-07-20).
"""
from __future__ import annotations

import numpy as np


def reshape_datacube(datacube: np.ndarray) -> np.ndarray:
    """datacube (nx, ny, K) → x_mat (M, K), M = nx*ny.  complex128 для точности."""
    nx, ny, k_snap = datacube.shape
    return datacube.reshape(nx * ny, k_snap).astype(np.complex128, copy=False)


def sample_covariance(x_mat: np.ndarray) -> np.ndarray:
    """Выборочная ковариация R = X @ Xᴴ / K.  x_mat — (M, K)."""
    k_snap = x_mat.shape[1]
    return (x_mat @ x_mat.conj().T) / k_snap


def apply_diagonal_loading(r_cov: np.ndarray, loading: float) -> np.ndarray:
    """Diagonal loading: R' = R + loading · (tr(R)/M) · I.  loading<=0 → без изменений.

    Масштаб tr(R)/M — средняя мощность на канал, чтобы loading был безразмерным
    и не зависел от абсолютного уровня сигнала (см. docstring `SubspaceNuller.loading`
    и `RobustMvdrNuller.loading`).
    """
    if loading <= 0.0:
        return r_cov
    m_elem = r_cov.shape[0]
    return r_cov + loading * (np.trace(r_cov).real / m_elem) * np.eye(m_elem, dtype=r_cov.dtype)
