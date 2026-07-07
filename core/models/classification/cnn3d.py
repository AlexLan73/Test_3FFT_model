"""Маленькая 3D-CNN классификатор куба (PyTorch, ROCm).

torch импортируется ЛЕНИВО -- пакет работает и без него (детерминированный
классификатор не зависит от torch). Сеть резидентна в VRAM, инференс батчем.
"""
from __future__ import annotations

import numpy as np

from ..result import SpectralCube
from .classifier import CubeClassifier
from .labels import CLASS_NAMES, Classification


def build_cnn3d(n_classes: int = len(CLASS_NAMES), in_ch: int = 1, width: int = 8):
    """Собирает 3D-CNN: 2 conv (stride 2) -> GAP -> FC. ~4 тыс. параметров."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv3d(in_ch, width, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(width, width * 2, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool3d(1),
        nn.Flatten(),
        nn.Linear(width * 2, n_classes),
    )


class Cnn3DClassifier(CubeClassifier):
    """Инференс-обёртка над обученной сетью."""

    def __init__(self, weights_path: str | None = None, device: str | None = None,
                 width: int = 8):
        import torch
        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net = build_cnn3d(width=width).to(self._device).eval()
        if weights_path:
            self._net.load_state_dict(torch.load(weights_path, map_location=self._device))

    def classify(self, cube: SpectralCube) -> Classification:
        torch = self._torch
        x = torch.from_numpy(cube.magnitude.astype("float32"))[None, None].to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._net(x), dim=1)[0].cpu().numpy()
        label = int(probs.argmax())
        energy = (cube.magnitude ** 2).sum(axis=2)
        ix, iy = np.unravel_index(int(np.argmax(energy)), energy.shape)
        return Classification(
            label, CLASS_NAMES[label], float(probs[label]),
            {n: float(p) for n, p in zip(CLASS_NAMES, probs, strict=True)},
            (float(cube.kx.values[ix]), float(cube.ky.values[iy])),
        )
