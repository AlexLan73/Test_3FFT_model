"""Детерминированный классификатор по форме отклика (работает без обучения).

Воплощает доказуемый различитель: точка / гребёнка / заливка / пусто.
Заградительную и стороннее излучение по одной форме надёжно не разделить
(оба заливают дальность) -- это и есть место, где помогает обучаемая сеть.
"""
from __future__ import annotations

import numpy as np

from ..result import SpectralCube
from .classifier import CubeClassifier
from .labels import CLASS_NAMES, Classification


class RuleBasedClassifier(CubeClassifier):
    def __init__(self, peak_snr: float = 4.0, fill_frac: float = 0.45,
                 peak_rel: float = 0.5):
        self._snr = peak_snr
        self._fill = fill_frac
        self._rel = peak_rel

    def classify(self, cube: SpectralCube) -> Classification:
        mag = cube.magnitude
        floor = float(np.median(mag)) + 1e-9
        energy = (mag ** 2).sum(axis=2)
        ix, iy = np.unravel_index(int(np.argmax(energy)), energy.shape)
        kx = float(cube.kx.values[ix])
        ky = float(cube.ky.values[iy])
        prof = mag[ix, iy, :]
        peak = float(prof.max())

        if peak < self._snr * floor:
            return self._make("empty", 0.7, kx, ky)

        occupancy = float(np.mean(prof > 0.3 * peak))
        n_peaks = self._count_peaks(prof, self._rel * peak)

        if occupancy > self._fill:                       # заливка по дальности
            name = "ham" if (abs(kx) + abs(ky) > 2) else "barrage"
            conf = 0.6                                   # форму разделить нельзя -> низкая уверенность
        elif n_peaks >= 2:
            name, conf = "comb", 0.8
        else:
            name, conf = "target", 0.85
        return self._make(name, conf, kx, ky)

    @staticmethod
    def _count_peaks(prof: np.ndarray, thr: float) -> int:
        above = prof > thr
        rising = np.sum(above[1:] & ~above[:-1])
        return int(rising) + int(above[0])

    @staticmethod
    def _make(name: str, conf: float, kx: float, ky: float) -> Classification:
        rest = (1.0 - conf) / (len(CLASS_NAMES) - 1)
        probs = {n: (conf if n == name else rest) for n in CLASS_NAMES}
        return Classification(CLASS_NAMES.index(name), name, conf, probs, (kx, ky))
