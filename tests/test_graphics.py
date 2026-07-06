"""Тесты графики radar3d (layout/sampler/scatter/interactive).

Запуск:  python tests/test_graphics.py
ВНИМАНИЕ: pytest ЗАПРЕЩЁН (см. .claude/rules/04-testing-python.md).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Чтобы работала форма `python tests/test_graphics.py` (не только `-m tests.test_graphics`).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.runner import AssertionGroup, SkipTest, TestRunner  # noqa: E402


class GraphicsTests(TestRunner):

    def setup(self) -> None:
        from core.config import default_scenario
        from core.controller import SimulationController
        from core.data_context import DataContext
        from core.models import AxisWindows, Fft3DModel, HannWindow

        cfg = default_scenario()
        model = Fft3DModel(cfg.array, cfg.range,
                           windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()))
        ctrl = SimulationController(model=model, data_context=DataContext(root="./out/data"))
        self.cube = ctrl.run(cfg, save_as="test_graphics_scene").spectral_cube

    def test_layout_axes(self) -> AssertionGroup:
        from core.graphics import AxisLayout

        g = AssertionGroup("graphics.layout")

        vertical = AxisLayout.range_vertical()
        g.add((vertical.axis_x, vertical.axis_y, vertical.axis_z) == ("kx", "ky", "range"),
              "range_vertical() -> (kx, ky, range)")

        depth = AxisLayout.range_in_depth()
        g.add((depth.axis_x, depth.axis_y, depth.axis_z) == ("kx", "range", "ky"),
              "range_in_depth() -> (kx, range, ky)")

        values, label, limits = vertical.resolve(self.cube, "kx")
        g.add("kx" in label, "label для kx содержит 'kx'")
        g.add(limits[0] <= float(values.min()) and limits[1] >= float(values.max()),
              "limits для центрированной оси охватывают min/max (с паддингом)")
        g.add(limits[1] - limits[0] > float(values.max() - values.min()),
              "паддинг для centered-оси действительно расширяет диапазон")

        rvalues, rlabel, rlimits = vertical.resolve(self.cube, "range")
        g.add("дальность" in rlabel, "label для range содержит 'дальность'")
        g.add(rlimits == (float(rvalues.min()), float(rvalues.max())),
              "limits для НЕ центрированной оси = ровно (min, max), без паддинга")
        return g

    def test_sampler_threshold(self) -> AssertionGroup:
        from core.graphics import AxisLayout, CubeSampler

        g = AssertionGroup("graphics.sampler")
        layout = AxisLayout.range_vertical()

        thr = -20.0
        sampler = CubeSampler(threshold_db=thr)
        pts = sampler.points(self.cube, layout)
        g.add(pts.mask.sum() == pts.x.size == pts.y.size == pts.z.size == pts.values_db.size,
              "число точек согласовано с маской по всем массивам")
        g.add(bool((pts.values_db > thr).all()), "все выбранные точки строго выше порога")

        rmax = 10
        sampler_cut = CubeSampler(threshold_db=-1000.0, range_limit=rmax)
        pts_cut = sampler_cut.points(self.cube, layout)
        nx, ny = self.cube.kx.values.size, self.cube.ky.values.size
        g.add(pts_cut.mask.size == nx * ny * rmax,
              "range_limit режет дальностную ось ДО построения маски (axis=2)")
        g.add(pts_cut.mask.all(), "порог -1000 дБ отбирает все точки среза")
        return g

    def test_scatter_default_regression(self) -> AssertionGroup:
        """Регрессия дефолтного вида -- сверка по набору точек/маске (не по байтам PNG, F5)."""
        from matplotlib.figure import Figure

        from core.graphics import AxisLayout, CubeSampler, CubeScatterVisualizer

        g = AssertionGroup("graphics.scatter_default_regression")

        thr, rmax = -22.0, 40
        vis = CubeScatterVisualizer(threshold_db=thr, range_limit=rmax)
        fig = vis.render(self.cube)
        g.add(isinstance(fig, Figure), "render() возвращает matplotlib.Figure")

        # прежняя (дорефакторинговая) формула отбора точек, воспроизведённая напрямую:
        m = self.cube.magnitude_db[:, :, :rmax]
        expected_mask_sum = int((m.ravel() > thr).sum())

        sampler = CubeSampler(threshold_db=thr, range_limit=rmax)
        pts = sampler.points(self.cube, AxisLayout.range_vertical())
        g.add(pts.x.size == expected_mask_sum,
              "CubeSampler даёт тот же набор точек, что и старая инлайн-формула")
        return g

    def test_scatter_range_in_depth(self) -> AssertionGroup:
        from matplotlib.figure import Figure

        from core.graphics import AxisLayout, CubeScatterVisualizer

        g = AssertionGroup("graphics.scatter_range_in_depth")
        vis = CubeScatterVisualizer(threshold_db=-22, range_limit=40,
                                    layout=AxisLayout.range_in_depth())
        fig = vis.render(self.cube)
        g.add(isinstance(fig, Figure), "новая раскладка тоже даёт валидную Figure")
        g.add(len(fig.axes) >= 1, "фигура содержит хотя бы один subplot")
        return g

    def test_interactive_default_guard(self) -> AssertionGroup:
        """F6: если default_db не входит в thresholds -- берём ближайший, без ValueError."""
        try:
            from core.graphics.interactive import InteractiveCubeVisualizer
        except ImportError as exc:
            raise SkipTest("plotly не установлен -- интерактивная ветка пропущена") from exc

        g = AssertionGroup("graphics.interactive_default_guard")
        thresholds = [-40, -30, -20, -10]
        vis = InteractiveCubeVisualizer(thresholds=thresholds, default_db=-22)
        g.add(vis.default_db == -20, "ближайший к -22 из [-40,-30,-20,-10] -- это -20")

        vis_ok = InteractiveCubeVisualizer(thresholds=thresholds, default_db=-30)
        g.add(vis_ok.default_db == -30, "default_db из набора остаётся без изменений")
        return g

    def test_interactive_html_written(self) -> AssertionGroup:
        try:
            from core.graphics.interactive import HtmlWriter, InteractiveCubeVisualizer
        except ImportError as exc:
            raise SkipTest("plotly не установлен -- интерактивная ветка пропущена") from exc

        g = AssertionGroup("graphics.interactive_html")
        vis = InteractiveCubeVisualizer()
        fig = vis.render(self.cube)

        out_dir = "./out/figures"
        writer = HtmlWriter(out_dir)
        path = writer.write(fig, "test_cube_interactive.html")
        g.add(os.path.exists(path), "HTML-файл создан")
        g.add(os.path.getsize(path) > 0, "HTML-файл не пуст")

        content = Path(path).read_text(encoding="utf-8")
        g.add("<div" in content, "в HTML есть <div (plotly-контейнер)")
        g.add("plotly" in content.lower(), "в HTML есть упоминание plotly (inline JS)")
        return g

    def test_matplotlib_branch_no_plotly_leak(self) -> AssertionGroup:
        """matplotlib-ветка не должна тянуть plotly при `import core.graphics` (F3)."""
        g = AssertionGroup("graphics.no_plotly_leak")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        result = subprocess.run(
            [sys.executable, "-c", "import core.graphics, sys; print('plotly' in sys.modules)"],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=60,
        )
        g.add(result.returncode == 0, f"подпроцесс отработал без ошибок: {result.stderr[-500:]}")
        g.add(result.stdout.strip() == "False",
              "plotly НЕ попадает в sys.modules при `import core.graphics`")
        return g


if __name__ == "__main__":
    GraphicsTests().run_all()
