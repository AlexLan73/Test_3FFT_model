"""Источники СЫРЬЯ (`RawCubeSource`, Strategy, спека §2.2/§3) -- транспорт, без обработки.

Реализации равнозначны (Strategy) -- все гонят `RawFrame` в `sink` (обычно
`RawQueue.put`), различаются ТОЛЬКО происхождением кадра:

- `SocketSource` -- сырьё генерит внешний GPU/C++-процесс, кадры идут по сети
  (ZMQ/`transport.py`). **Этап C -- ВНЕ этой задачи** (нужен ZMQ/GPU-процесс,
  другой чат); здесь оставлена только эта заметка, класс НЕ реализован.
- `FileSource` -- оффлайн-реплей записанного набора с диска (Этап D, этот модуль).

Сама обработка сырья (`SignalFrontend`/детекция/tick) -- ВНЕ этого модуля (Этап B).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Event

import numpy as np

from .raw_queue import RawFrame


class RawCubeSource(ABC):
    """Strategy -- абстрактный источник сырых кадров (§2.2 спеки).

    Реализации равнозначны для потребителя: любая из них гонит `RawFrame` в `sink`
    (обычно `RawQueue.put`), пока не выставлен `stop`. Различие -- ТОЛЬКО в
    происхождении кадра (сеть/GPU/диск), тракт обработки после `sink` -- один (§3).
    """

    @abstractmethod
    def run(self, sink: Callable[[RawFrame], None], stop: Event) -> None:
        """Гнать кадры в `sink`, пока не выставлен `stop` (или пока не кончился набор)."""


class FileSource(RawCubeSource):
    """Оффлайн-реплей записанного набора сырья с диска (Этап D, §3 спеки).

    Формат набора: файлы `*.npy` в `directory`, **отсортированные по имени** = такты
    по порядку; каждый файл -- один сырой куб (`np.load`). `index` кадра -- порядковый
    номер файла в отсортированном списке (не парсится из имени). `sig` -- единый тип
    сигнала на весь набор (передаётся параметром, в самих файлах не хранится).
    """

    def __init__(self, directory: str | Path, sig: str = "lfm", delay_s: float = 0.0) -> None:
        self._directory = Path(directory)
        self._sig = sig
        self._delay_s = delay_s

    def _sorted_files(self) -> list[Path]:
        """Отсортированный по имени снапшот `*.npy`-файлов набора."""
        return sorted(self._directory.glob("*.npy"))

    def iter_frames(self) -> Iterator[RawFrame]:
        """Чистый (без сети/sleep/stop) итератор кадров -- для тестов и разбора набора."""
        for i, path in enumerate(self._sorted_files()):
            cube = np.load(path)
            yield RawFrame(index=i, cube=cube, sig=self._sig)

    def run(self, sink: Callable[[RawFrame], None], stop: Event) -> None:
        """Реалтайм-имитация: отдать кадры набора в `sink` по порядку с паузой `delay_s`.

        Прерывается, если `stop` выставлен ДО начала (0 кадров) или между кадрами.
        """
        for frame in self.iter_frames():
            if stop.is_set():
                return
            sink(frame)
            if self._delay_s > 0.0:
                time.sleep(self._delay_s)


__all__ = ["RawCubeSource", "FileSource"]
