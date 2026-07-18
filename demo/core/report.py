"""DemoReport — Value Object со сводкой одного прогона примера (§3.5 спеки).

Frozen dataclass: собирается один раз в конце `DemoRunner.run()`, дальше только
читается (тестами-приёмкой и консольным выводом `demo/run_all.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DemoReport:
    """Метрики прогона примера (Value Object, неизменяемый после создания).

    `classification`/`verdicts`/`decisions` — задел под ex2+ (куб-ветка, токенизатор,
    арбитр); для ex1 остаются дефолтами (пусто/None) — DemoRunner их не наполняет.
    """

    example: str
    figures: list[str]
    n_tokens: int = 0
    verdicts: tuple[Any, ...] = ()
    decisions: tuple[Any, ...] = ()
    classification: Any | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"DemoReport[{self.example}]: {len(self.figures)} фигур"]
        if self.n_tokens:
            parts.append(f"{self.n_tokens} токенов")
        if self.verdicts:
            parts.append(f"вердикты={self.verdicts}")
        if self.decisions:
            parts.append(f"решения={self.decisions}")
        if self.classification is not None:
            parts.append(f"класс={self.classification}")
        if self.metrics:
            kv = ", ".join(f"{k}={v}" for k, v in self.metrics.items())
            parts.append(f"metrics: {kv}")
        return " · ".join(parts)
