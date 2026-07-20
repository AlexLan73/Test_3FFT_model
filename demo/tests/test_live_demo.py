"""Приёмка `demo/ex4_flight/live_demo.py` -- живой демо-источник (Этап A, 🚫 pytest, правило 04).

Проверяет `build_live_tick` -- ЧИСТУЮ функцию (без сети/sleep): контракт §2.1 (truth/pts/trk/sl),
след трека не длиннее `kTrail`, все значения -- примитивы (не numpy), соседние такты дают
РАЗНЫЕ позиции цели ("живой" поток), и roundtrip через `core.runtime.codec` (то, что реально
уйдёт в сокет).

Запуск:  .venv/bin/python demo/tests/test_live_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.runtime import codec  # noqa: E402
from demo.ex4_flight.live_demo import (  # noqa: E402
    LiveSceneConfig,
    build_live_meta,
    build_live_tick,
)


def _is_primitive(value: object) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_primitive(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_primitive(v) for k, v in value.items())
    return False


class LiveDemoTests(TestRunner):
    """`build_live_tick`/`build_live_meta` -- чистые функции, контракт §2.1 спеки."""

    def setup(self) -> None:
        self.cfg = LiveSceneConfig()

    def test_tick_contract_shape(self) -> AssertionGroup:
        g = AssertionGroup("live_demo.tick_shape")
        tick = build_live_tick(5, self.cfg)
        g.add("truth" in tick and "pts" in tick and "trk" in tick, "такт содержит truth/pts/trk")
        g.add(set(tick["truth"].keys()) == {"t", "c", "b"}, "truth содержит t/c/b")
        g.add(len(tick["truth"]["t"]) == 3, "truth.t -- [kx,ky,r]")
        g.add(len(tick["truth"]["b"]) == 2, "truth.b -- [kx,ky] (без дальности)")
        g.add(tick["sl"] == [], "срезов без куба нет -- sl пуст (панель терпит)")
        g.add(tick["band"] is None, "полосы barrage нет -- band=null")
        g.add(len(tick["pts"]) >= 1, "хотя бы детекция самой цели присутствует")
        g.add(len(tick["trk"]) == 1 and tick["trk"][0]["id"] == 1, "один трек id=1")
        return g

    def test_track_history_bounded_by_ktrail(self) -> AssertionGroup:
        g = AssertionGroup("live_demo.trail_bound")
        for step in (0, 1, 3, self.cfg.k_trail, self.cfg.k_trail + 20, 300):
            tick = build_live_tick(step, self.cfg)
            h = tick["trk"][0]["h"]
            g.add(len(h) <= self.cfg.k_trail,
                  f"шаг {step}: история трека должна быть <= kTrail={self.cfg.k_trail}, получили {len(h)}")
            g.add(len(h) == min(step + 1, self.cfg.k_trail),
                  f"шаг {step}: длина истории должна расти до kTrail, получили {len(h)}")
        return g

    def test_all_values_are_primitives(self) -> AssertionGroup:
        """Панель/codec не понимают numpy -- все значения такта должны быть примитивами."""
        g = AssertionGroup("live_demo.primitives_only")
        tick = build_live_tick(11, self.cfg)
        g.add(_is_primitive(tick), "весь payload такта должен состоять из примитивов (не numpy)")
        return g

    def test_adjacent_ticks_have_different_positions(self) -> AssertionGroup:
        """Поток должен быть "живым" -- соседние такты двигают цель, а не стоят на месте."""
        g = AssertionGroup("live_demo.moving_target")
        positions = [tuple(build_live_tick(step, self.cfg)["truth"]["t"]) for step in range(6)]
        g.add(len(set(positions)) > 1, f"позиции цели должны различаться по тактам: {positions}")
        for i in range(len(positions) - 1):
            a, b = positions[i], positions[i + 1]
            g.add(a != b, f"соседние такты не должны совпадать: {a} vs {b}")
        return g

    def test_deterministic_for_same_step(self) -> AssertionGroup:
        """Детерминизм: один и тот же step всегда даёт один и тот же такт (seeded rng, без random/time)."""
        g = AssertionGroup("live_demo.deterministic")
        t1 = build_live_tick(42, self.cfg)
        t2 = build_live_tick(42, self.cfg)
        g.add(t1 == t2, "build_live_tick(42, cfg) должен быть детерминирован")
        return g

    def test_meta_contract(self) -> AssertionGroup:
        g = AssertionGroup("live_demo.meta")
        meta = build_live_meta(self.cfg, n_ticks=None)
        g.add(meta["nx"] == self.cfg.nx and meta["ny"] == self.cfg.ny, "апертура в мете")
        g.add(meta["kTrail"] == self.cfg.k_trail, "kTrail в мете")
        g.add(meta["nTicks"] is None, "nTicks=None -- бесконечный поток")
        g.add(len(meta["stations"]) >= 1, "станции переиспользованы из server.STATIONS")
        g.add("cam" in meta and "field" in meta["cam"] and "scene" in meta["cam"],
              "камера в мете (единая модель core.graphics.Projection)")
        return g

    def test_codec_roundtrip(self) -> AssertionGroup:
        """То, что реально уйдёт в сокет: encode->decode переживает meta/tick без потерь."""
        g = AssertionGroup("live_demo.codec_roundtrip")
        meta = build_live_meta(self.cfg, n_ticks=10)
        raw_meta = codec.encode("meta", 0, meta)
        topic, _tact, back_meta = codec.decode(raw_meta)
        g.add(topic == "meta" and back_meta["nx"] == self.cfg.nx, "meta переживает codec roundtrip")

        tick = build_live_tick(2, self.cfg)
        raw_tick = codec.encode("tick", 2, tick)
        topic, tact, back_tick = codec.decode(raw_tick)
        g.add(topic == "tick" and tact == 2, "tick topic/tact сохранены")
        g.add(back_tick["trk"][0]["h"] == tick["trk"][0]["h"], "история трека переживает roundtrip")
        return g


if __name__ == "__main__":
    sys.exit(0 if LiveDemoTests().run_all() else 1)
