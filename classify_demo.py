"""Демо классификатора без torch: генерируем по кубу на класс и узнаём их.

Показывает фабрику датасета + детерминированный классификатор end-to-end.
"""
from __future__ import annotations
import numpy as np

from core.config import ArrayConfig, RangeConfig
from core.models import Fft3DModel, SpectralCube, Axis
from core.models.classification import (CubeDatasetGenerator, RuleBasedClassifier,
                                        CLASS_NAMES)


def _cube_from(magnitude, array, rng):
    kx = Axis("kx", np.arange(-array.nx // 2, array.nx // 2), True)
    ky = Axis("ky", np.arange(-array.ny // 2, array.ny // 2), True)
    rr = Axis("range", np.arange(rng.n_fft), False)
    return SpectralCube(magnitude, kx, ky, rr)


def main() -> None:
    array, rng = ArrayConfig(16, 16), RangeConfig(16, 64)
    model = Fft3DModel(array, rng)
    gen = CubeDatasetGenerator(array, rng, model, seed=1)
    clf = RuleBasedClassifier()

    print("истина     -> предсказание")
    correct = 0
    for name in CLASS_NAMES:
        mag, _ = gen.sample(name)
        result = clf.classify(_cube_from(mag, array, rng))
        ok = "OK" if result.name == name else "  "
        correct += result.name == name
        print(f"{ok} {name:10s} -> {result}")
    print(f"\nверно (детерминир.): {correct}/{len(CLASS_NAMES)}")


if __name__ == "__main__":
    main()
