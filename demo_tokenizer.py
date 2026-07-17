"""Демо токенизатора: подтверждение разделения классов по 6 признакам патента (§4.11).

Строит 3 угловые карты nx×ny (по умолчанию 16×16; цель / заград / шум) с Хэмминг-аподизацией (§4.3),
считает `FeatureExtractor` (§4.5) и рисует карты + таблицу признаков против
патентных коридоров. Это визуал-аналог `feat_scene.png` главы 4.

Выход:
    graphics/body_motion/p_tokenizer/feat_scene.png

Запуск:
    python demo_tokenizer.py
    python demo_tokenizer.py --seed 7
    python demo_tokenizer.py --nx 6 --ny 15   # неквадратная апертура i×j (E5), паддинг до 2ⁿ
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.config import ArrayConfig  # noqa: E402
from core.graphics import FigureWriter  # noqa: E402
from core.models.tokenizer import FeatureExtractor, RuleBasedTriage  # noqa: E402

OUT_DIR = os.path.join("graphics", "body_motion", "p_tokenizer")

_BG = "#0d1117"
_FG = "#c9d1d9"

# Патентные коридоры §4.11 (для подписи-сверки).
_PATENT = {
    "цель": dict(PR=3.6, Hoyer=0.94, MainFrac=0.98, LobeRatio=0.002, MaxMean=123),
    "заград": dict(PR=19.0, Hoyer=0.81, MainFrac=0.40, LobeRatio=0.25, MaxMean=32),
    "шум": dict(PR=129.0, Hoyer=0.31, MainFrac=0.07, LobeRatio=1.03, MaxMean=5.4),
}


def _steer(kx: float, ky: float, nx: int, ny: int) -> np.ndarray:
    """Плоская волна на апертуре nx×ny (углы kx/ky в циклах решётки)."""
    x = np.arange(nx)
    y = np.arange(ny)
    return np.outer(np.exp(1j * 2 * np.pi * kx * x / nx),
                    np.exp(1j * 2 * np.pi * ky * y / ny))


def _angular_power(aperture: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """P = |FFT2(апертура · Хэмминг)|² с fftshift (§4.3-4.4).

    Zero-pad до 2ⁿ по каждой угловой оси независимо (`ArrayConfig.padded_shape()`,
    §E5/F9) -- при дефолтных nx=ny=16 паддинг no-op.
    """
    window = np.outer(np.hamming(nx), np.hamming(ny))
    pow2x, pow2y = ArrayConfig(nx, ny).padded_shape()
    spectrum = np.fft.fftshift(np.fft.fft2(aperture * window, s=(pow2x, pow2y)))
    return np.abs(spectrum) ** 2


def _scenes(rng: np.random.Generator, nx: int, ny: int) -> dict[str, np.ndarray]:
    shape = (nx, ny)
    target = _steer(3, -2, nx, ny) + 0.02 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    barrage = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    for _ in range(8):
        barrage = barrage + 3.0 * _steer(rng.uniform(-7, 7), rng.uniform(-7, 7), nx, ny)
    noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return {"цель": target, "заград": barrage, "шум": noise}


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо токенизатора: разделение классов по §4.11.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nx", type=int, default=16, help="число элементов решётки по X (дефолт 16)")
    parser.add_argument("--ny", type=int, default=16, help="число элементов решётки по Y (дефолт 16)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    extractor = FeatureExtractor()
    triage = RuleBasedTriage()
    scenes = _scenes(rng, args.nx, args.ny)
    pow2x, pow2y = ArrayConfig(args.nx, args.ny).padded_shape()
    print(f"Апертура: nx×ny={args.nx}×{args.ny}, padded={(pow2x, pow2y)}")

    plt.rcParams.update({
        "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
        "text.color": _FG, "axes.edgecolor": _FG, "axes.labelcolor": _FG,
        "xtick.color": _FG, "ytick.color": _FG,
    })

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    fig.suptitle("Токенизатор §4.5: 6 признаков разделяют цель / заград / шум (сверка с §4.11)",
                 color=_FG, fontsize=13)

    for col, (name, aperture) in enumerate(scenes.items()):
        power = _angular_power(aperture, args.nx, args.ny)
        f = extractor.extract(power)
        label, score = triage.classify(f)

        ax = axes[0, col]
        ax.imshow(10.0 * np.log10(power / power.max() + 1e-12), cmap="magma",
                  origin="lower", vmin=-40, vmax=0)
        ax.set_title(f"{name}: угловая карта P (дБ)\nтриаж -> «{label}» (скор {score:.2f})",
                     color=_FG, fontsize=10)
        ax.set_xlabel("kx")
        ax.set_ylabel("ky")

        ax2 = axes[1, col]
        ax2.axis("off")
        got = dict(PR=f.pr, Hoyer=f.hoyer, MainFrac=f.main_frac,
                   LobeRatio=f.lobe_ratio, MaxMean=f.max_mean)
        pat = _PATENT[name]
        rows = [f"{'признак':<10}{'наш':>9}{'патент':>10}"]
        rows.append("-" * 29)
        for k in ("PR", "Hoyer", "MainFrac", "LobeRatio", "MaxMean"):
            rows.append(f"{k:<10}{got[k]:>9.3f}{pat[k]:>10.3g}")
        ax2.text(0.05, 0.95, "\n".join(rows), family="monospace", fontsize=11,
                 color=_FG, va="top", transform=ax2.transAxes)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = FigureWriter(OUT_DIR).write(fig, "feat_scene.png")
    plt.close(fig)
    print(f"PNG записан: {path}")


if __name__ == "__main__":
    main()
