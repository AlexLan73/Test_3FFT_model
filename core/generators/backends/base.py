"""GenBackend — настоящий Strategy-протокол бэкенда генерации (§4.3 спеки, P1).

Заменяет пустую заглушку `core/generators/waveforms/base.py` (P0). НЕ импортируется
оттуда напрямую в рантайме (во избежание цикла waveforms↔backends) — только через
`TYPE_CHECKING` в сигнатурах `Waveform.render`, там же `from __future__ import
annotations` уже делает аннотацию строковой.

Две реализации (DIP/LSP, взаимозаменяемы):
    NumpyBackend — эталон/Windows/датасет (P1, этот файл + numpy_backend.py).
    HipBackend   — боевой GPU (DSP-GPU `.so`, P2, ещё не создан).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class GenBackend(Protocol):
    """Примитивы генерации на NumPy **или** HIP — один интерфейс, Waveform их не различает."""

    def exp_phase(self, phase: np.ndarray) -> np.ndarray:
        """exp(1j·phase) → complex64[...]."""
        ...

    def apply_window(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """x·mask (вне маски — 0). Не мутирует `x`."""
        ...

    def add_noise(self, x: np.ndarray, power: float,
                   rng: np.random.Generator) -> np.ndarray:
        """x + комплексный AWGN мощности `power` (CN(0, power)). Не мутирует `x`."""
        ...
