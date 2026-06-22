"""Обучение 3D-CNN на синтетических кубах (PyTorch-ROCm).

Запуск на машине с torch+ROCm:
    python train_cnn.py --steps 400 --batch 40
Данные генерируются на лету фабрикой CubeDatasetGenerator (метки идеальны).
"""
from __future__ import annotations
import argparse

from core.config import ArrayConfig, RangeConfig
from core.models import Fft3DModel
from core.models.classification import (CubeDatasetGenerator, build_cnn3d,
                                        CLASS_NAMES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=str, default="cnn3d.pt")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    array, rng = ArrayConfig(16, 16), RangeConfig(16, 64)
    model = Fft3DModel(array, rng)
    gen = CubeDatasetGenerator(array, rng, model, seed=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = build_cnn3d().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    print(f"device={device}, классы={CLASS_NAMES}, "
          f"параметров={sum(p.numel() for p in net.parameters())}")

    net.train()
    for step in range(1, args.steps + 1):
        X, y = gen.batch(args.batch)
        X = torch.from_numpy(X).to(device)
        y = torch.from_numpy(y).to(device)
        opt.zero_grad()
        out = net(X)
        loss = lossf(out, y)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            acc = (out.argmax(1) == y).float().mean().item()
            print(f"step {step:4d}  loss {loss.item():.3f}  acc {acc:.2f}")

    torch.save(net.state_dict(), args.out)
    print("сохранено:", args.out)


if __name__ == "__main__":
    main()
