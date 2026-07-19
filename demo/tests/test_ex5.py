"""Приёмка ex5 (парабола в кубе) через `common.runner.TestRunner` (🚫 pytest, правило 04).

Лёгкие размеры: 16×16×64, 2 объекта — полный 32×32×256 гоняет сам demo-прогон.

Запуск:  .venv/Scripts/python.exe demo/tests/test_ex5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работала форма `python demo/tests/test_ex5.py` (конвенция репо).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import ArrayConfig, ProjectConfig  # noqa: E402
from core.generators.waveforms import AmToCube  # noqa: E402
from demo.ex5_peak_refine.example import (  # noqa: E402
    Ex5Params,
    Ex5PeakRefine,
    FracObject,
    analyze_cube,
    build_clean_volume,
    top_peaks,
    true_index,
)

_TEST_SCENE = (
    FracObject("X", kx=-3.60, ky=2.30, freq_hz=100.30e6),
    FracObject("Y", kx=4.25, ky=-5.70, freq_hz=140.85e6),
)

_TEST_PARAMS = Ex5Params(
    nx=16, ny=16, n_axis=64, depth=64, fs=500e6,
    snr_db_list=(float("inf"), 0.0), seed=7, scene=_TEST_SCENE,
)


class Ex5PeakRefineTests(TestRunner):

    def setup(self) -> None:
        self.p = _TEST_PARAMS
        self.cfg = ProjectConfig(array=ArrayConfig(self.p.nx, self.p.ny), modulation="am")
        rng = np.random.default_rng(self.p.seed)
        volume = build_clean_volume(self.p, rng)
        self.cube = AmToCube(depth=self.p.depth, step=32).fill(volume, self.cfg)

    def test_volume_shape(self) -> AssertionGroup:
        g = AssertionGroup("ex5.volume_shape")
        rng = np.random.default_rng(self.p.seed)
        volume = build_clean_volume(self.p, rng)
        g.add(volume.shape == (self.p.nx, self.p.ny, self.p.n_axis),
              f"shape должен быть {(self.p.nx, self.p.ny, self.p.n_axis)}, получено {volume.shape}")
        g.add(volume.dtype == np.complex64, f"dtype должен быть complex64, получено {volume.dtype}")
        return g

    def test_top_peaks_nms(self) -> AssertionGroup:
        g = AssertionGroup("ex5.top_peaks")
        power = self.cube.magnitude.astype(np.float64) ** 2
        peaks = top_peaks(power, 2, self.p.guard)
        g.add(len(peaks) == 2, f"NMS должен найти 2 пика, получено {len(peaks)}")
        for obj in self.p.scene:
            t = true_index(self.p, obj)
            near = min(abs(pk[0] - t[0]) + abs(pk[1] - t[1]) for pk in peaks)
            g.add(near < 1.5, f"объект {obj.name}: ближайший пик в {near:.2f} бинах (>1.5)")
        return g

    def test_refine_beats_argmax(self) -> AssertionGroup:
        """Главная приёмка: парабола точнее argmax по каждой оси каждого объекта."""
        g = AssertionGroup("ex5.refine_beats_argmax")
        results = analyze_cube(self.cube, self.p)
        g.add(len(results) == 2, f"сопоставлено объектов {len(results)}, ожидалось 2")
        for r in results:
            for ax in range(3):
                e_ref, e_arg = r.err_refined[ax], r.err_argmax[ax]
                g.add(e_ref < 0.15,
                      f"{r.obj.name} ось {ax}: ошибка параболы {e_ref:.3f} ≥ 0.15 бина")
                g.add(e_ref <= e_arg + 1e-9,
                      f"{r.obj.name} ось {ax}: парабола ({e_ref:.3f}) хуже argmax ({e_arg:.3f})")
        return g

    def test_runner_report(self) -> AssertionGroup:
        """Полный прогон Template Method без записи PNG: метрики по обоим SNR на месте."""
        g = AssertionGroup("ex5.runner_report")
        report = Ex5PeakRefine(_TEST_PARAMS).run(save=False)
        g.add(report.example == "ex5_peak_refine", "имя примера в отчёте")
        g.add("clean" in report.metrics and "snr+0" in report.metrics,
              f"метрики обоих SNR должны присутствовать, получено {sorted(report.metrics)}")
        g.add(all("3/3" in m or "2/2" in m for m in report.metrics.values()),
              f"все объекты должны сопоставиться: {report.metrics}")
        return g


def _web_params():
    from demo.ex5_peak_refine.web import Ex5WebParams
    # разнос целей под маленькую апертуру 16×16 (дефолтные ±9 упёрлись бы в клип края)
    return Ex5WebParams(nx=16, ny=16, n_axis=64, depth=64, n_ticks=5,
                        snr_db_list=(float("inf"), 0.0), seed=7,
                        offsets=((-4.0, 3.0), (0.5, -4.5), (4.5, 0.5)))


class Ex5WebTests(TestRunner):
    """Приёмка живого web-прототипа (маленькие размеры: 16×16×64, 5 тактов)."""

    def test_trajectories_move(self) -> AssertionGroup:
        from demo.ex5_peak_refine.web import build_trajectories
        g = AssertionGroup("ex5web.trajectories")
        p = _web_params()
        traj = build_trajectories(p)
        g.add(len(traj) == p.n_ticks, f"тактов {len(traj)}, ожидалось {p.n_ticks}")
        g.add(all(len(scene) == len(p.names) for scene in traj), "в каждом такте все цели")
        moved = max(abs(traj[-1][k].kx - traj[0][k].kx) + abs(traj[-1][k].ky - traj[0][k].ky)
                    for k in range(len(p.names)))
        g.add(moved > 0.05, f"цели должны двигаться, макс. смещение {moved:.3f} бина")
        in_field = all(abs(o.kx) <= p.nx / 2 and abs(o.ky) <= p.ny / 2
                       for scene in traj for o in scene)
        g.add(in_field, "все позиции в пределах поля")
        return g

    def test_simulate_refine_beats_argmax(self) -> AssertionGroup:
        from demo.ex5_peak_refine.web import simulate
        g = AssertionGroup("ex5web.simulate")
        p = _web_params()
        data = simulate(p)
        g.add(set(data["runs"]) == {"clean", "snr+0"},
              f"оба SNR-прогона: {sorted(data['runs'])}")
        for tag, run in data["runs"].items():
            g.add(all(len(t["tg"]) == len(p.names) for t in run["ticks"]),
                  f"{tag}: все цели сопоставлены в каждом такте")
            e_arg = [abs(q["am"][ax] - q["tr"][ax])
                     for t in run["ticks"] for q in t["tg"] for ax in range(3)]
            e_ref = [abs(q["pr"][ax] - q["tr"][ax])
                     for t in run["ticks"] for q in t["tg"] for ax in range(3)]
            m_arg, m_ref = float(np.mean(e_arg)), float(np.mean(e_ref))
            g.add(m_ref < m_arg, f"{tag}: парабола ({m_ref:.3f}) хуже argmax ({m_arg:.3f})")
            g.add(m_ref < 0.1, f"{tag}: средняя ошибка параболы {m_ref:.3f} ≥ 0.1 бина")
        return g

    def test_build_page(self) -> AssertionGroup:
        import json
        import re
        import tempfile

        from demo.ex5_peak_refine.web import build_page, simulate
        g = AssertionGroup("ex5web.page")
        p = _web_params()
        data = simulate(p)
        with tempfile.TemporaryDirectory() as tmp:
            path = build_page(data, Path(tmp) / "index.html")
            html = path.read_text(encoding="utf-8")
        g.add("__DATA__" not in html, "плейсхолдер данных должен быть заменён")
        g.add("<canvas" in html, "страница должна содержать canvas")
        g.add("https://" not in html and "http://" not in html,
              "страница самодостаточна: никаких внешних URL (офлайн-требование)")
        m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
        g.add(m is not None, "встроенный JSON должен находиться регэкспом")
        if m is not None:
            parsed = json.loads(m.group(1))
            g.add(parsed["nTicks"] == p.n_ticks, "nTicks в данных страницы")
        return g


if __name__ == "__main__":
    ok = Ex5PeakRefineTests().run_all()
    ok = Ex5WebTests().run_all() and ok
    sys.exit(0 if ok else 1)
