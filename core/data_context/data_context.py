"""Фасад доступа к данным: единая точка входа для load/save + рантайм-шина."""
from __future__ import annotations

import numpy as np

from .message_bus import MessageBus, Observer
from .repository import CubeRepository, NpyCubeRepository


class DataContext:
    """Facade над репозиториями + композиция `MessageBus` (F1, SRP).

    Персистентность (`save_cube`/`load_cube`, обратная совместимость -- сигнатура
    не меняется) и рантайм pub/sub (`publish`/`subscribe`) -- разные ответственности:
    шина не встроена сюда, а композируется отдельным объектом `MessageBus`.
    """

    def __init__(self, repository: CubeRepository | None = None,
                 root: str = "./data", bus: MessageBus | None = None):
        self._repo = repository or NpyCubeRepository(root)
        self._bus = bus or MessageBus()

    def save_cube(self, name: str, cube: np.ndarray) -> str:
        return self._repo.save(name, cube)

    def load_cube(self, name: str) -> np.ndarray:
        return self._repo.load(name)

    @property
    def bus(self) -> MessageBus:
        """Рантайм-шина (Subject). Прямого I/O в обход `DataContext`/шины быть не должно (§4)."""
        return self._bus

    def publish(self, key: str, data: object) -> None:
        """Удобство: делегирует в `self.bus.publish` (кладёт данные + уведомляет наблюдателей)."""
        self._bus.publish(key, data)

    def subscribe(self, key: str, obs: Observer) -> None:
        """Удобство: делегирует в `self.bus.subscribe`."""
        self._bus.subscribe(key, obs)
