"""RawQueue -- план СЫРЬЯ (спека `realtime_panel_2026-07-19.md`, §4.1/§7).

Два разных плана данных панели НЕ путать (§4.1):

| План           | Что                         | Размер        | Хранение                              |
|----------------|------------------------------|---------------|----------------------------------------|
| **Сырьё**      | сырой куб с GPU/SSD          | велико (~МБ..ГБ/такт) | **транзит: consume-and-drop**  |
| Обработанное   | лёгкий `tick` (`TickLog`)     | ~6 КБ/такт    | храним ВСЁ                             |

Этот модуль -- ТОЛЬКО первый план (сырьё). `RawQueue` -- ограниченная очередь между
источником сырья (продюсер, отдельный тред/процесс-приёмник) и обработкой (консьюмер,
Этап B, другой чат): память не копим (кадр может весить единицы-десятки ГБ), при
переполнении дропаем самый старый кадр (как PUB slow-joiner, см. `transport.py` N1) --
консьюмер получает самое свежее, а не застрявшую историю сырья.

Сама обработка сырья (`SignalFrontend`/ядро детекции) -- ВНЕ этого модуля (Этап B).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RawFrame:
    """Value Object -- один сырой кадр (СЫРЬЁ, не путать с лёгким `Tick`, §4.1).

    `cube` -- сырой комплексный объём по всем антеннам (сигнал+шум+помехи), форма
    nx*ny*N, dtype complex64 (или что прислал источник -- эта VO НЕ проверяет и НЕ
    приводит тип/форму, это забота обработки, Этап B). `sig` -- тип сигнала
    ("lfm"/"am"/...) -- по нему обработка выбирает `SignalFrontend` (§3 Этап B).

    Кадр живёт ТРАНЗИТОМ в `RawQueue`: обработка забирает `get()` и роняет ссылку --
    сырьё в очереди не копится (велико, до ~10 ГБ/такт в худшем случае, §4.5).
    """

    index: int
    cube: np.ndarray
    sig: str


class RawQueue:
    """Pure Fabrication -- ограниченная очередь СЫРЬЯ, consume-and-drop (§4.1).

    Продюсер (источник, `RawCubeSource`) и консьюмер (обработка, Этап B) живут в
    разных тредах -- доступ потокобезопасен (`threading.Condition`). При переполнении
    `put()` **дропает самый старый** кадр (FIFO consume-and-drop, аналог PUB
    slow-joiner из `transport.py`): память не растёт, консьюмер всегда получает
    самые свежие такты, а не застрявшую историю большого сырья.

    `on_drop` -- опциональная точка расширения (Pure Fabrication hook): вызывается
    для КАЖДОГО выброшенного кадра (например, архив блоками на диск для повторного
    прогона, §4.1). Сам архивирующий хук в этой задаче НЕ реализуется -- только
    контракт вызова.
    """

    def __init__(self, maxsize: int = 4, on_drop: Callable[[RawFrame], None] | None = None) -> None:
        if maxsize < 1:
            raise ValueError(f"maxsize должен быть >= 1, получено {maxsize}")
        self._maxsize = maxsize
        self._on_drop = on_drop
        self._frames: list[RawFrame] = []
        self._dropped = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    @property
    def dropped(self) -> int:
        """Счётчик кадров, выброшенных из-за переполнения (consume-and-drop)."""
        with self._lock:
            return self._dropped

    def put(self, frame: RawFrame) -> None:
        """Положить кадр в очередь; при переполнении дропнуть самый старый (FIFO)."""
        dropped_frame: RawFrame | None = None
        with self._not_empty:
            if len(self._frames) >= self._maxsize:
                dropped_frame = self._frames.pop(0)
                self._dropped += 1
            self._frames.append(frame)
            self._not_empty.notify()
        if dropped_frame is not None and self._on_drop is not None:
            self._on_drop(dropped_frame)

    def get(self, timeout: float | None = None) -> RawFrame | None:
        """Забрать самый старый кадр (FIFO); `None`, если очередь пуста по таймауту.

        Ждёт в цикле по монотонному дедлайну (`time.monotonic()`), а не одним
        `wait(timeout)` — иначе ложное пробуждение (spurious wakeup) вернуло бы
        `None` раньше фактического истечения `timeout`.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_empty:
            while not self._frames:
                if deadline is None:
                    self._not_empty.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(timeout=remaining)
            return self._frames.pop(0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


__all__ = ["RawFrame", "RawQueue"]
