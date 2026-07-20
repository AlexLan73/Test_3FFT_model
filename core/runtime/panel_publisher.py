"""PanelPublisher -- тонкий публикатор панели поверх `Transport` (Этап A, спека §4.1).

Развязывает ИСТОЧНИК данных (реалтайм-цикл/генератор) от транспорта: источник вызывает
`push_meta`/`push_tick`, публикатор кладёт такт в `TickLog` (Pure Fabrication, "храним ВСЁ",
§4.1) и публикует его через любую реализацию `Transport` (DI -- зависимость от Protocol,
не от `WebSocketTransport`, см. `core/runtime/transport.py`).

`WebSocketTransport` publish-only и без `on_connect` (см. докстринг класса) -- поздний
браузер узнаёт `meta` только при повторной публикации. `republish_meta()` -- ручной хук под
это (демо-цикл зовёт периодически, см. `demo/ex4_flight/live_demo.py`). Полноценный
`on_connect`-реплей (meta + весь лог сессии позднему клиенту) -- вне объёма Этапа A, оставлен
как TODO (§4.1 п.4 спеки: "остаётся реализовать хук on_connect").
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .transport import Transport


@dataclass(frozen=True, slots=True)
class Tick:
    """Value Object -- один иммутабельный лёгкий такт панели (контракт §2.1 спеки).

    `payload` -- уже-примитивный словарь (`truth`/`band`/`pts`/`trk`/`sl`/`feats`),
    готовый к `codec.encode` без дополнительной обработки.
    """

    index: int
    payload: dict[str, Any]


class TickLog:
    """Pure Fabrication -- append-only лог ОБРАБОТАННЫХ тактов ("храним ВСЁ", §4.1).

    Только хранит такты и отдаёт неизменяемый снапшот -- без транспорта/сети внутри
    (SRP). Опциональный `cap` -- предохранитель-потолок (FIFO-дроп самого старого),
    по умолчанию `None` -- ничего не режем (лёгкий tick ~6 КБ, 1000 тактов ~6 МБ, §4.1).
    """

    def __init__(self, cap: int | None = None) -> None:
        self._ticks: deque[Tick] = deque(maxlen=cap)

    def append(self, tick: Tick) -> None:
        """Добавить такт в конец лога (дропает самый старый при переполнении `cap`)."""
        self._ticks.append(tick)

    def snapshot(self) -> tuple[Tick, ...]:
        """Неизменяемая копия текущего лога (для позднего клиента/перемотки)."""
        return tuple(self._ticks)

    def __len__(self) -> int:
        return len(self._ticks)


class PanelPublisher:
    """Тонкая обёртка над `Transport` (DI, §1/§4.1 спеки) -- публикует meta/tick панели.

    Зависит ТОЛЬКО от абстракции `Transport` (Protocol), не от конкретной реализации
    (`WebSocketTransport`/`ZmqTransport`/`FanOutTransport` -- любая подходит).
    """

    def __init__(self, transport: Transport, log: TickLog | None = None) -> None:
        self._transport = transport
        self._log = log if log is not None else TickLog()
        self._meta: dict[str, Any] | None = None

    @property
    def log(self) -> TickLog:
        """Доступ к логу тактов (для тестов/перемотки)."""
        return self._log

    def start(self) -> None:
        """Запустить транспорт, если у него есть `start()` (напр. `WebSocketTransport`)."""
        start = getattr(self._transport, "start", None)
        if callable(start):
            start()

    def push_meta(self, meta: dict[str, Any]) -> None:
        """Опубликовать `meta` сессии и запомнить её для `republish_meta()`."""
        self._meta = meta
        self._transport.publish("meta", 0, meta)

    def push_tick(self, index: int, payload: dict[str, Any]) -> None:
        """Записать такт в `TickLog` и опубликовать его через транспорт."""
        self._log.append(Tick(index, payload))
        self._transport.publish("tick", index, payload)

    def republish_meta(self) -> None:
        """Повторно опубликовать сохранённую `meta` (поздний клиент, см. докстринг модуля).

        TODO (§4.4, вне Этапа A): заменить на `on_connect`-хук транспорта -- реплей
        `meta` + весь лог сессии (или хвост) конкретному подключившемуся клиенту, а не
        широковещательный повтор всем.
        """
        if self._meta is not None:
            self._transport.publish("meta", 0, self._meta)

    def close(self) -> None:
        """Закрыть транспорт, если у него есть `close()`."""
        close = getattr(self._transport, "close", None)
        if callable(close):
            close()
