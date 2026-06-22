"""Базовая модель РЛС-обработки (Strategy для приложения, Template Method внутри)."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

from .result import SpectralCube


class RadarModel(ABC):
    """Преобразует сырой куб данных в SpectralCube.

    Шаблонный метод process() фиксирует скелет (окно -> преобразование -> упаковка);
    подклассы переопределяют _transform/_build_axes.
    """

    def process(self, datacube: np.ndarray) -> SpectralCube:
        windowed = self._apply_windows(datacube)
        spectrum = self._transform(windowed)
        return self._build_result(spectrum)

    @abstractmethod
    def _apply_windows(self, cube: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _transform(self, cube: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def _build_result(self, spectrum: np.ndarray) -> SpectralCube: ...
