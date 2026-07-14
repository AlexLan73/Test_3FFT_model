"""WaveTimeConfig — расширенный конфиг сырого времени для генераторов сигналов (P0, §5.1 спеки).

Реюзит существующий `ArrayConfig` (nx, ny) — геометрию решётки не дублируем.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .array_config import ArrayConfig


@dataclass(frozen=True)
class WaveTimeConfig:
    """Baseline-параметры сырого времени (числа §5.1: патент/industrial/DSP-GPU defaults)."""

    fs: float = 12e6              # дискретизация, Гц (DSP-GPU factories default)
    carrier_hz: float = 2e6       # несущая/IF, Гц (DSP-GPU default)
    fdev_hz: float = 6e6          # полоса ЛЧМ / девиация, Гц (патент гл.7.5 → Δr≈25 м)
    n_samples: int = 8192         # длина по быстрому времени (DSP-GPU default)
    array: ArrayConfig = field(default_factory=lambda: ArrayConfig(nx=16, ny=16))
    seed: int = 7                 # rng seed (R6: детерминизм)
