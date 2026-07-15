# vendored from GPUWorkLib/PyPanelAntennas/SNR/dechirp_numpy.py
#            + DSP-GPU/DSP/Python/heterodyne/heterodyne_base.py (2026-07-15)
"""Гетеродин / дечирп ЛЧМ (вендор из DSP-GPU / GPUWorkLib).

Дечирп = умножение принятого сигнала на **комплексно-сопряжённый опорный ЛЧМ**:

    s_dc(t) = s_rx(t) · conj(s_ref(t))

После дечирпа задержка цели `tau = 2R/c` превращается в постоянную beat-частоту
`f_b = chirp_rate · tau` (тон), и дальность извлекается дальностным FFT (пик на `f_b`).
Эталон GPU: `HeterodyneDechirp` (ядро `dechirp_multiply`, DSP-GPU/heterodyne).
Эталон numpy: `dechirp_numpy.dechirp` (GPUWorkLib) / `HeterodyneTestBase.dechirp_numpy` (DSP-GPU).
"""
from __future__ import annotations

import numpy as np


def dechirp(rx: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """`s_dc = rx · conj(ref)`. `ref` (N,) бродкастится по последней оси `rx` (..., N)."""
    return (rx * np.conj(ref)).astype(np.complex64)
