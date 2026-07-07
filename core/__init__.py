"""core -- ядро модели радиолокационной обработки.

Слои:
    config/        -- неизменяемые конфигурации и спецификации сцены
    generators/    -- источники сигналов, помехи, сцена (Composite), фабрика
    models/        -- модели обработки (сейчас 3D-БПФ), окна, результат
    graphics/      -- визуализаторы и запись фигур
    data_context/  -- загрузка/выгрузка данных
    controller.py  -- координатор прогона
"""
from .controller import ProcessingOutcome, SimulationController

__all__ = ["SimulationController", "ProcessingOutcome"]
