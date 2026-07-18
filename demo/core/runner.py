"""DemoRunner — Template Method общего прогона примера (§3.1 спеки).

Скелет неизменен для всех примеров: собрать контекст → прогнать hook-шаги в
фиксированном порядке → сохранить фигуры → собрать `DemoReport`. Каждый пример
переопределяет **только** нужные ему hook-методы (ex1 — `build_signal` +
`visualize`; ex2+ — остальные, добавляются по мере постановки задач).
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from core.generators.waveforms import SignalField

from .report import DemoReport
from .writer import DemoWriter


@dataclass(frozen=True)
class DemoContext:
    """Состояние одного прогона, прокидывается между hook-шагами (SRP)."""

    name: str
    cfg: object
    rng: np.random.Generator


class DemoRunner(ABC):
    """Базовый класс примера. `run()` — Template Method, подклассы её не трогают."""

    name: str = ""
    seed: int = 7

    # ── hook-методы: базовый класс возвращает None/пусто, подкласс переопределяет ──

    def build_signal(self, ctx: DemoContext) -> SignalField | None:
        """ex1: собрать 1D `SignalField` — эталон для отчёта. По умолчанию — None."""
        return None

    def build_volume(self, ctx: DemoContext) -> np.ndarray | None:
        """ex2+: собрать объём `[nx,ny,N]` complex64. По умолчанию — None."""
        return None

    def to_cube(self, ctx: DemoContext, volume: np.ndarray) -> Any | None:
        """ex2+: объём → `SpectralCube` (3FFT/AmToCube/LfmToCube). По умолчанию — None."""
        return None

    def tokenize(self, ctx: DemoContext, cube: Any) -> tuple[list, list] | None:
        """ex3+: `SpectralCube` → `(tokens, verdicts)`. По умолчанию — None."""
        return None

    def arbitrate(self, ctx: DemoContext, verdicts: list) -> list | None:
        """ex3+: вердикты → `list[TargetDecision]`. По умолчанию — None."""
        return None

    def classify(self, ctx: DemoContext, cube: Any) -> Any | None:
        """ex2+: `SpectralCube` → `Classification`. По умолчанию — None."""
        return None

    def visualize(self, ctx: DemoContext) -> dict[str, Figure]:
        """Собрать все PNG-фигуры примера. По умолчанию — пусто (нет графиков)."""
        return {}

    def report_metrics(self, ctx: DemoContext) -> dict[str, Any]:
        """Доп. числа для `DemoReport.metrics` (SNR, пики, число отсчётов...). По умолчанию — пусто."""
        return {}

    # ── Template Method — не переопределять в подклассах ──

    def run(self, *, save: bool = True) -> DemoReport:
        """Прогнать пример целиком: контекст → hook-шаги → (опц.) запись PNG → отчёт."""
        ctx = DemoContext(name=self.name, cfg=None, rng=np.random.default_rng(self.seed))

        # build_signal выполняется для ex1 (эталонный SignalField отчёта); его результат
        # пока не прокидывается дальше по шагам — цепочка signal→volume для ex2+ появится
        # вместе с SceneBank (см. §3.2 общей спеки), сейчас не изобретаем заранее.
        self.build_signal(ctx)
        volume = self.build_volume(ctx)
        cube = self.to_cube(ctx, volume) if volume is not None else None
        tokens_verdicts = self.tokenize(ctx, cube) if cube is not None else None
        tokens, verdicts = tokens_verdicts if tokens_verdicts is not None else ([], [])
        decisions = self.arbitrate(ctx, verdicts) if verdicts else None
        classification = self.classify(ctx, cube) if cube is not None else None

        figures = self.visualize(ctx)

        paths: list[str] = []
        if save and figures:
            writer = DemoWriter(self.name)
            for fig_name, fig in figures.items():
                paths.append(writer.write(fig, f"{fig_name}.png"))

        return DemoReport(
            example=self.name,
            figures=paths,
            n_tokens=len(tokens),
            verdicts=tuple(verdicts),
            decisions=tuple(decisions) if decisions else (),
            classification=classification,
            metrics=self.report_metrics(ctx),
        )
