"""Приёмка ex4-СЕРВЕРА — стример живой панели по сокету (🚫 pytest, правило 04).

Заменил test_ex4_web.py (самодостаточная HTML-панель → архив). Проверяет сериализацию
такта/меты в примитивы (`server.tick_payload`/`meta_payload`), кроп срезов ±8, финальные
признаки §4.11, roundtrip через `core.runtime.codec` (то, что реально уходит в сокет) и
единую камеру `Projection` (поле развёрнуто относительно 3D — вид с нулевой дальности).

Полный размер 64×64×4096 (решение Alex 1A), но 3 такта — историю гоняем один раз на класс.

Запуск:  .venv/Scripts/python.exe demo/tests/test_ex4_server.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.graphics import Projection  # noqa: E402
from core.runtime import codec  # noqa: E402
from demo.ex4_flight.example import Ex4Flight, Ex4Params  # noqa: E402
from demo.ex4_flight.server import (  # noqa: E402
    CROP_HALF,
    build_session,
    meta_payload,
    tick_payload,
)


class Ex4ServerTests(TestRunner):
    """Один прогон истории на класс (кеш) + сериализация примитивов + сокет-контракт."""

    _cached: Ex4Flight | None = None

    def setup(self) -> None:
        if Ex4ServerTests._cached is None:
            ex = Ex4Flight(params=Ex4Params(tacts=3))
            ex.run_history(np.random.default_rng(ex.seed))
            Ex4ServerTests._cached = ex
        self.ex = Ex4ServerTests._cached
        self.meta, self.ticks = build_session(self.ex)

    def test_meta_payload(self) -> AssertionGroup:
        g = AssertionGroup("ex4srv.meta")
        m = self.meta
        g.add(m["nx"] == 64 and m["ny"] == 64 and m["nAxis"] == 4096, "апертура/ось в мете")
        g.add(m["nTicks"] == 3, f"nTicks={m['nTicks']}, ожидалось 3")
        g.add(len(m["stations"]) == 2, "две стационарные помехи-станции")
        g.add("cam" in m and "field" in m["cam"] and "scene" in m["cam"], "параметры камеры в мете")
        g.add(abs(m["cam"]["field"]["az"] - np.pi) < 1e-9,
              "field-камера смотрит С НУЛЕВОЙ дальности (az=π)")
        return g

    def test_tick_payload_world_coords(self) -> AssertionGroup:
        """Такт → примитивы (МИРОВЫЕ kx,ky,r); проекция — на клиенте единой камерой."""
        g = AssertionGroup("ex4srv.tick")
        g.add(len(self.ticks) == 3, "все такты сериализованы")
        t = self.ticks[-1]
        g.add(len(t["pts"]) > 0, "точки-детекции такта присутствуют")
        g.add(t["band"] is not None, "полоса barrage в данных")
        g.add(len(t["trk"]) > 0, "треки в данных")
        g.add(all(len(tr["h"]) <= self.meta["kTrail"] for tr in t["trk"]),
              f"история трека обрезана до K={self.meta['kTrail']}")
        g.add(all(isinstance(p[0], (int, float)) for p in t["pts"]),
              "точки — примитивы (не numpy)")
        return g

    def test_slices_cropped(self) -> AssertionGroup:
        """Решение Alex 2A: срезы — кроп ±8 бинов, не полный nx×ny."""
        g = AssertionGroup("ex4srv.crop")
        w = 2 * CROP_HALF + 1
        slices = [s for t in self.ticks for s in t["sl"]]
        g.add(len(slices) > 0, "срезы присутствуют (треки возраста >=2)")
        for s in slices:
            g.add(len(s["m"]) == w and all(len(row) == w for row in s["m"]),
                  f"кроп №{s['id']} должен быть {w}×{w}")
        return g

    def test_final_features(self) -> AssertionGroup:
        """Признаки §4.11 — финального такта (канон ex4)."""
        g = AssertionGroup("ex4srv.final_feats")
        ff = self.meta["finalFeats"]
        g.add(len(ff) > 0, "финальные признаки хотя бы одного трека")
        need = {"pr", "hoyer", "main_frac", "lobe_ratio", "max_mean", "energy"}
        for tid, rec in ff.items():
            g.add(need.issubset(rec["f"].keys()),
                  f"трек {tid}: все 6 признаков §4.11")
        return g

    def test_codec_roundtrip(self) -> AssertionGroup:
        """То, что реально уходит в сокет: encode→decode (msgpack, как читает web/msgpack.js)."""
        g = AssertionGroup("ex4srv.codec")
        raw_meta = codec.encode("meta", 0, self.meta)
        topic, _tact, back = codec.decode(raw_meta)
        g.add(topic == "meta" and back["nTicks"] == 3, "meta сериализуется через codec без потерь")
        raw_tick = codec.encode("tick", 2, self.ticks[-1])
        topic, tact, back = codec.decode(raw_tick)
        g.add(topic == "tick" and tact == 2, "tick topic/tact сохранены")
        g.add(len(back["pts"]) == len(self.ticks[-1]["pts"]), "точки такта переживают roundtrip")
        return g

    def test_field_flipped_vs_scene(self) -> AssertionGroup:
        """Сквозная проверка: поле развёрнуто по дальности (+kx вправо), 3D-куб +kx влево."""
        g = AssertionGroup("ex4srv.field_flip")
        scene = Projection(nx=64, ny=64, n_range=4096)
        field = Projection.field(64, 64, 4096)
        g.add(field.project(20, 0, 0)[0] > field.project(-20, 0, 0)[0],
              "поле: +kx вправо (вид с нулевой дальности)")
        g.add(scene.project(20, 0, 2000)[0] < scene.project(-20, 0, 2000)[0],
              "3D-куб: +kx влево (облётный ракурс)")
        return g


if __name__ == "__main__":
    sys.exit(0 if Ex4ServerTests().run_all() else 1)
