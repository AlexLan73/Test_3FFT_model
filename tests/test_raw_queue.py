"""Тесты плана СЫРЬЯ (`core.runtime.raw_queue`/`raw_source`, спека `realtime_panel_2026-07-19.md` §4.1/§2.2).

`RawQueueTests` -- `RawQueue` consume-and-drop (FIFO, дроп самого старого, `on_drop`-хук,
потокобезопасность продюсер/консьюмер). `FileSourceTests` -- `FileSource` (Этап D,
оффлайн-реплей набора `*.npy`), `iter_frames`/`run`.

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_raw_queue.py
"""
from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402


def _cube(index: int) -> np.ndarray:
    """Маленький сырой куб-заглушка с меткой такта в [0,0,0] для сверки порядка."""
    c = np.zeros((4, 4, 8), dtype=np.complex64)
    c[0, 0, 0] = index
    return c


class RawQueueTests(TestRunner):
    """`RawQueue` -- ограниченная очередь СЫРЬЯ, consume-and-drop (§4.1)."""

    def test_put_get_fifo_order(self) -> AssertionGroup:
        g = AssertionGroup("raw_queue.fifo_order")
        from core.runtime.raw_queue import RawFrame, RawQueue

        q = RawQueue(maxsize=10)
        for i in range(3):
            q.put(RawFrame(index=i, cube=_cube(i), sig="lfm"))
        out = [q.get(timeout=1.0).index for _ in range(3)]
        g.add(out == [0, 1, 2], f"get() должен отдавать кадры в порядке put (FIFO), получено {out}")
        return g

    def test_get_empty_returns_none_on_timeout(self) -> AssertionGroup:
        g = AssertionGroup("raw_queue.get_empty_timeout")
        from core.runtime.raw_queue import RawQueue

        q = RawQueue(maxsize=2)
        result = q.get(timeout=0.05)
        g.add(result is None, f"get() на пустой очереди по таймауту должен вернуть None, получено {result}")
        return g

    def test_consume_and_drop_keeps_freshest(self) -> AssertionGroup:
        g = AssertionGroup("raw_queue.consume_and_drop")
        from core.runtime.raw_queue import RawFrame, RawQueue

        q = RawQueue(maxsize=2)
        for i in range(5):
            q.put(RawFrame(index=i, cube=_cube(i), sig="lfm"))
        g.add(len(q) <= 2, f"maxsize=2 -- длина не должна превышать потолок, len={len(q)}")
        g.add(q.dropped == 3, f"5 put при maxsize=2 -> 3 дропа, получено {q.dropped}")

        remaining = []
        while True:
            fr = q.get(timeout=0.05)
            if fr is None:
                break
            remaining.append(fr.index)
        g.add(remaining == [3, 4], f"после дропа должны остаться самые СВЕЖИЕ кадры, получено {remaining}")
        return g

    def test_on_drop_hook_called_for_each_dropped_frame(self) -> AssertionGroup:
        g = AssertionGroup("raw_queue.on_drop_hook")
        from core.runtime.raw_queue import RawFrame, RawQueue

        dropped_indices: list[int] = []
        q = RawQueue(maxsize=2, on_drop=lambda fr: dropped_indices.append(fr.index))
        for i in range(5):
            q.put(RawFrame(index=i, cube=_cube(i), sig="lfm"))
        g.add(dropped_indices == [0, 1, 2],
              f"on_drop должен получить каждый выброшенный кадр по порядку, получено {dropped_indices}")
        return g

    def test_producer_consumer_threads_no_hang(self) -> AssertionGroup:
        """Продюсер-тред кладёт N кадров, потребитель-тред читает -- без зависания."""
        g = AssertionGroup("raw_queue.producer_consumer_threads")
        from core.runtime.raw_queue import RawFrame, RawQueue

        n = 20
        q = RawQueue(maxsize=4)
        received: list[int] = []
        consumer_done = threading.Event()

        def produce() -> None:
            for i in range(n):
                q.put(RawFrame(index=i, cube=_cube(i), sig="lfm"))

        def consume() -> None:
            empty_gets = 0
            while True:
                fr = q.get(timeout=0.5)
                if fr is None:
                    empty_gets += 1
                    if empty_gets > 4:
                        break
                    continue
                received.append(fr.index)
                if fr.index == n - 1:
                    break
            consumer_done.set()

        producer = threading.Thread(target=produce)
        consumer = threading.Thread(target=consume)
        consumer.start()
        producer.start()
        producer.join(timeout=5.0)
        consumer.join(timeout=5.0)

        g.add(consumer_done.is_set(), "потребитель-тред должен завершиться (не зависнуть)")
        g.add(not producer.is_alive() and not consumer.is_alive(), "оба треда должны завершиться в таймаут")
        g.add(received == sorted(received), f"полученные индексы должны идти по возрастанию, {received}")
        g.add(len(received) > 0, "потребитель должен получить хотя бы часть кадров")
        return g


class FileSourceTests(TestRunner):
    """`FileSource` -- оффлайн-реплей набора `*.npy` (Этап D, §3 спеки)."""

    def setup(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._dir = Path(self._tmp.name)
        self._cubes = []
        for i in range(3):
            c = np.zeros((4, 4, 8), dtype=np.complex64)
            c[0, 0, 0] = i
            self._cubes.append(c)
            np.save(self._dir / f"frame_{i:03d}.npy", c)

    def test_iter_frames_yields_in_order_with_correct_meta(self) -> AssertionGroup:
        g = AssertionGroup("raw_source.iter_frames_order")
        from core.runtime.raw_source import FileSource

        src = FileSource(self._dir, sig="lfm")
        frames = list(src.iter_frames())
        g.add(len(frames) == 3, f"должно быть ровно 3 кадра, получено {len(frames)}")
        g.add([f.index for f in frames] == [0, 1, 2], f"index должен идти по порядку, {[f.index for f in frames]}")
        g.add(all(f.sig == "lfm" for f in frames), "sig должен быть единым на набор")
        g.add(all(bool(np.array_equal(f.cube, self._cubes[i])) for i, f in enumerate(frames)),
              "кубы должны совпасть с записанными на диск")
        return g

    def test_run_sends_all_frames_to_sink_in_order(self) -> AssertionGroup:
        g = AssertionGroup("raw_source.run_sink_order")
        from core.runtime.raw_source import FileSource

        src = FileSource(self._dir, sig="am", delay_s=0.0)
        received = []
        stop = threading.Event()
        src.run(received.append, stop)
        g.add([f.index for f in received] == [0, 1, 2], f"run() должен отдать 3 кадра по порядку, {received}")
        g.add(all(f.sig == "am" for f in received), "sig должен прийти в каждом кадре")
        return g

    def test_run_stop_before_start_yields_zero_frames(self) -> AssertionGroup:
        g = AssertionGroup("raw_source.run_stop_before_start")
        from core.runtime.raw_source import FileSource

        src = FileSource(self._dir, sig="lfm")
        received = []
        stop = threading.Event()
        stop.set()
        src.run(received.append, stop)
        g.add(received == [], f"stop до старта должен дать 0 кадров, получено {len(received)}")
        return g


if __name__ == "__main__":
    ok = True
    for cls in (RawQueueTests, FileSourceTests):
        ok = cls().run_all() and ok
    sys.exit(0 if ok else 1)
