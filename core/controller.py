"""Координатор сценария обработки (GRASP Controller).

Зависит только от абстракций (DIP): синтезатор, модель, репозиторий передаются
извне. Контроллер не знает о конкретных источниках/окнах/форматах хранения.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import SimulationConfig
from .data_context import DataContext
from .generators import SceneBuilder, Synthesizer
from .models import RadarModel


@dataclass
class ProcessingOutcome:
    """Результат прогона: сырой куб и спектральный куб."""
    raw_cube: object
    spectral_cube: object


class SimulationController:
    def __init__(self, model: RadarModel,
                 scene_builder: SceneBuilder | None = None,
                 data_context: DataContext | None = None):
        self._model = model
        self._builder = scene_builder or SceneBuilder()
        self._data = data_context

    def run(self, cfg: SimulationConfig, save_as: str | None = None) -> ProcessingOutcome:
        scene = self._builder.build(cfg.scene)
        synth = Synthesizer(cfg.array, cfg.range, cfg.seed)
        raw = synth.build(scene)
        spectral = self._model.process(raw)
        if save_as and self._data is not None:
            self._data.save_cube(save_as, raw)
        return ProcessingOutcome(raw_cube=raw, spectral_cube=spectral)
