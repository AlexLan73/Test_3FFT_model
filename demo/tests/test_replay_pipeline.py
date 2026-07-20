"""Приёмка `demo/ex4_flight/replay_pipeline.py` -- сквозное демо реалтайм-проводки БЕЗ GPU
(🚫 pytest, правило 04).

Проверяет: `write_demo_dataset` пишет набор `*.npy`, `process_dataset` (чистая функция без
сети/тредов) даёт валидный контракт §2.1 на каждый такт, заглушка `naive_cube_to_tick` ловит
внедрённый пик (argmax находит именно вставленный «объект», а не шум), и сквозная проводка
через `RawQueue` (producer-тред + consumer) отдаёт такты по порядку в fake-транспорт БЕЗ
реального WS/GPU.

Запуск:  .venv/bin/python demo/tests/test_replay_pipeline.py
"""
from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.runtime import FileSource, PanelPublisher, RawQueue  # noqa: E402
from demo.ex4_flight.replay_pipeline import (  # noqa: E402
    _object_cell,
    naive_cube_to_tick,
    process_dataset,
    write_demo_dataset,
)


def _is_primitive(value: object) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_primitive(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_primitive(v) for k, v in value.items())
    return False


class _FakeTransport:
    """Fake `Transport` (Protocol) для теста проводки -- без реального WS-сокета/сети."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, int, dict[str, Any]]] = []

    def publish(self, topic: str, tact: int, payload: object) -> None:
        self.messages.append((topic, tact, payload))  # type: ignore[arg-type]

    def subscribe(self, topic: str, callback: object) -> None:
        raise NotImplementedError("fake транспорт -- publish-only, как WebSocketTransport")


class ReplayPipelineTests(TestRunner):
    """`write_demo_dataset` / `process_dataset` / заглушка / сквозная проводка через `RawQueue`."""

    NX, NY, N_AXIS = 8, 8, 32

    def setup(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)

    def test_write_demo_dataset_creates_files(self) -> AssertionGroup:
        g = AssertionGroup("replay.write_dataset")
        paths = write_demo_dataset(self.directory, n_ticks=5, nx=self.NX, ny=self.NY,
                                    n_axis=self.N_AXIS)
        g.add(len(paths) == 5, f"должно быть 5 файлов, получили {len(paths)}")
        g.add(all(p.exists() for p in paths), "все файлы набора должны существовать на диске")
        g.add(paths == sorted(paths), "пути должны быть отсортированы по имени = порядку тактов")
        self._tmp.cleanup()
        return g

    def test_process_dataset_valid_contract(self) -> AssertionGroup:
        g = AssertionGroup("replay.process_dataset")
        write_demo_dataset(self.directory, n_ticks=5, nx=self.NX, ny=self.NY, n_axis=self.N_AXIS)
        ticks = process_dataset(self.directory, sig="lfm", nx=self.NX, ny=self.NY)
        g.add(len(ticks) == 5, f"должно быть 5 tick, получили {len(ticks)}")
        for i, tick in enumerate(ticks):
            g.add(tick["truth"] is None, f"такт {i}: truth=None (заглушка)")
            g.add(tick["band"] is None, f"такт {i}: band=None (заглушка)")
            g.add(tick["sl"] == [], f"такт {i}: sl=[] (заглушка)")
            g.add(len(tick["pts"]) == 1, f"такт {i}: ровно одна точка-детекция")
            g.add(len(tick["pts"][0]) == 4, f"такт {i}: точка -- [kx,ky,pos,db]")
            g.add(len(tick["trk"]) == 1 and tick["trk"][0]["id"] == 1,
                  f"такт {i}: один трек id=1")
            g.add(_is_primitive(tick), f"такт {i}: все значения должны быть примитивами (не numpy)")
        self._tmp.cleanup()
        return g

    def test_stub_finds_injected_peak(self) -> AssertionGroup:
        """Заглушка должна найти именно ВСТАВЛЕННЫЙ объект, а не случайный шумовой максимум."""
        g = AssertionGroup("replay.stub_finds_peak")
        write_demo_dataset(self.directory, n_ticks=2, nx=self.NX, ny=self.NY, n_axis=self.N_AXIS)
        source = FileSource(self.directory, sig="lfm")
        frames = list(source.iter_frames())
        for step, frame in enumerate(frames):
            ix, iy, ir = _object_cell(step, self.NX, self.NY, self.N_AXIS)
            expected_kx = ix - self.NX // 2
            expected_ky = iy - self.NY // 2
            tick = naive_cube_to_tick(frame, self.NX, self.NY)
            kx, ky, pos, _db = tick["pts"][0]
            g.add(kx == expected_kx and ky == expected_ky,
                  f"такт {step}: детекция ({kx},{ky}) должна совпасть с вставленным объектом "
                  f"({expected_kx},{expected_ky})")
            g.add(pos == ir, f"такт {step}: дальность детекции {pos} должна быть {ir}")
        self._tmp.cleanup()
        return g

    def test_pipeline_wiring_through_raw_queue(self) -> AssertionGroup:
        """Сквозная проводка producer(тред)->RawQueue->consumer->fake-транспорт, БЕЗ реального WS/GPU."""
        g = AssertionGroup("replay.wiring")
        n_ticks = 5
        write_demo_dataset(self.directory, n_ticks=n_ticks, nx=self.NX, ny=self.NY,
                            n_axis=self.N_AXIS)

        raw_queue = RawQueue(maxsize=4)
        source = FileSource(self.directory, sig="lfm", delay_s=0.0)
        stop = threading.Event()
        producer = threading.Thread(target=source.run, args=(raw_queue.put, stop), daemon=True)

        fake_transport = _FakeTransport()
        publisher = PanelPublisher(fake_transport)  # type: ignore[arg-type]
        publisher.push_meta({"nx": self.NX, "ny": self.NY, "nAxis": self.N_AXIS})

        producer.start()
        received = 0
        while received < n_ticks:
            frame = raw_queue.get(timeout=2.0)
            if frame is None:
                if not producer.is_alive():
                    break
                continue
            publisher.push_tick(frame.index, naive_cube_to_tick(frame, self.NX, self.NY))
            received += 1
        stop.set()
        producer.join(timeout=2.0)

        g.add(received == n_ticks, f"consumer должен получить {n_ticks} тактов, получил {received}")
        tick_msgs = [m for m in fake_transport.messages if m[0] == "tick"]
        g.add(len(tick_msgs) == n_ticks, f"fake-транспорт должен получить {n_ticks} tick-сообщений")
        indices = [tact for _topic, tact, _payload in tick_msgs]
        g.add(indices == list(range(n_ticks)), f"индексы тактов должны идти по порядку 0..{n_ticks - 1}: {indices}")
        meta_msgs = [m for m in fake_transport.messages if m[0] == "meta"]
        g.add(len(meta_msgs) == 1, "meta должна быть опубликована один раз")
        self._tmp.cleanup()
        return g


if __name__ == "__main__":
    sys.exit(0 if ReplayPipelineTests().run_all() else 1)
