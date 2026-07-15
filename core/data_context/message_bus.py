"""MessageBus -- рантайм pub/sub-шина (Observer/Subject, F1).

Отдельно от `DataContext` (SRP, SPEC §3/§4): персистентность (Repository:
save_cube/load_cube) и рантайм-обмен между продюсером и наблюдателями (панель,
plotly, matplotlib) -- разные ответственности. `DataContext` шину композирует,
не наследует и не смешивает с I/O.

Синхронный notify (без потоков/очереди -- тредовый приём заложен в P6/S6, не P1).
Ключи-каналы (SPEC §4): "cube", "squares", "tracks", "config".
"""
from __future__ import annotations

from collections import defaultdict
from typing import Protocol


class Observer(Protocol):
    """Наблюдатель шины: получает данные, опубликованные под ключом `key`."""

    def on_data(self, key: str, data: object) -> None: ...


class MessageBus:
    """Subject (Observer-паттерн): subscribe/unsubscribe/publish -> notify всех подписчиков key."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Observer]] = defaultdict(list)

    def subscribe(self, key: str, obs: Observer) -> None:
        subs = self._subscribers[key]
        if obs not in subs:
            subs.append(obs)

    def unsubscribe(self, key: str, obs: Observer) -> None:
        subs = self._subscribers.get(key)
        if subs and obs in subs:
            subs.remove(obs)

    def publish(self, key: str, data: object) -> None:
        """Кладёт данные в шину и синхронно уведомляет всех подписчиков `key`."""
        for obs in list(self._subscribers.get(key, ())):
            obs.on_data(key, data)
