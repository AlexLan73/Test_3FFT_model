"""Приёмка ex4: полёт + барьер + анимация (🚫 pytest, правило 04).

Alex (спека §4): размер НАШ — 64×64×4096, но тактов 6 (время). GIF/plotly — SkipTest.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, SkipTest, TestRunner  # noqa: E402
from demo.ex4_flight.example import Ex4Flight, Ex4Params  # noqa: E402

_P = Ex4Params(tacts=6)          # 64×64×4096 (решение Alex), 6 тактов


def _run() -> Ex4Flight:
    ex = Ex4Flight(params=_P)
    ex.run(save=False)
    return ex


class Ex4FlightTests(TestRunner):
    """Один тяжёлый прогон на класс (кешируем) + проверки по истории."""

    _cached: Ex4Flight | None = None

    def setup(self) -> None:
        if Ex4FlightTests._cached is None:
            Ex4FlightTests._cached = _run()
        self.ex = Ex4FlightTests._cached

    def test_determinism(self) -> AssertionGroup:
        g = AssertionGroup("ex4.determinism")
        ex2 = Ex4Flight(params=Ex4Params(tacts=2))
        ex2.run(save=False)
        ex3 = Ex4Flight(params=Ex4Params(tacts=2))
        ex3.run(save=False)
        t2 = [r.truth["target"] for r in ex2._history]
        t3 = [r.truth["target"] for r in ex3._history]
        g.add(t2 == t3, "два прогона с одним seed дают одну траекторию (вся случайность от rng)")
        return g

    def test_target_moves(self) -> AssertionGroup:
        g = AssertionGroup("ex4.moves")
        kxs = [r.truth["target"][0] for r in self.ex._history]
        g.add(len(set(round(k, 3) for k in kxs)) > 1,
              f"kx цели меняется по тактам: {[round(k, 1) for k in kxs]}")
        return g

    def test_barrage_banded_each_tact(self) -> AssertionGroup:
        g = AssertionGroup("ex4.band")
        banded = [r.banded for r in self.ex._history]
        g.add(all(banded), f"полоса барьера детектирована на каждом такте: {banded}")
        return g

    def test_target_found_most_tacts(self) -> AssertionGroup:
        g = AssertionGroup("ex4.found")
        found = self.ex._stats["target_found"]
        n = int(found.split("/")[0])
        g.add(n >= len(self.ex._history) - 1,
              f"цель найдена >= {len(self.ex._history) - 1} тактов, получено {found}")
        return g

    def test_track_is_moving(self) -> AssertionGroup:
        g = AssertionGroup("ex4.track")
        tracks = self.ex._history[-1].tracks
        g.add(len(tracks) >= 1, f"есть треки, получено {len(tracks)}")
        g.add(any(t["is_moving"] and len(t["history"]) >= 3 for t in tracks),
              "есть устойчивый движущийся трек (>=3 тактов, is_moving)")
        return g

    def test_gifs_written(self) -> AssertionGroup:
        g = AssertionGroup("ex4.gifs")
        gifs = self.ex._stats.get("gifs", {})
        if not gifs:
            raise SkipTest("GIF не записаны (нет Pillow-writer)")
        for tag, path in gifs.items():
            p = Path(path)
            g.add(p.exists() and p.stat().st_size > 0, f"{tag}: файл есть и не пуст ({path})")
        return g


if __name__ == "__main__":
    Ex4FlightTests().run_all()
