"""Таксономия классов и результат классификации (Value Object)."""
from __future__ import annotations

from dataclasses import dataclass

# Классы отклика ячейки. empty -- пусто (только шум).
CLASS_NAMES: tuple[str, ...] = ("empty", "target", "barrage", "comb", "ham")


@dataclass(frozen=True)
class Classification:
    """Решение классификатора по одному кубу."""
    label: int
    name: str
    confidence: float
    probabilities: dict[str, float]
    cell: tuple[float, float]          # доминирующая угловая ячейка (kx, ky)

    def __str__(self) -> str:
        return (f"{self.name} (p={self.confidence:.2f}) "
                f"@ угол ({self.cell[0]:+.0f}, {self.cell[1]:+.0f})")
