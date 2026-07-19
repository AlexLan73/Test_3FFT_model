"""Тесты единой сцены/камеры (🚫 pytest, правило 04).

ДОКАЗЫВАЮТ: правое окно (поле) — та же сцена вдоль оси дальности (ОДИН `project`), поэтому
окна согласованы, зеркальность невозможна. Прецедент — `.claude/rules/07-math-in-core.md`.

Запуск:  .venv/Scripts/python.exe tests/test_camera.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.graphics import Projection  # noqa: E402


class CameraTests(TestRunner):

    def setup(self) -> None:
        self.scene = Projection(nx=64, ny=64, n_range=4096)
        self.field = Projection.field(64, 64, 4096)

    def test_field_is_scene_along_range(self) -> AssertionGroup:
        """Поле = та же сцена вдоль дальности: экран НЕ зависит от r (дальность схлопнута)."""
        g = AssertionGroup("camera.field_along_range")
        a = self.field.project(15, -8, 0)
        b = self.field.project(15, -8, 4096)
        g.add(abs(a[0]-b[0]) < 1e-12 and abs(a[1]-b[1]) < 1e-12,
              "поле: экранные x,y не должны зависеть от дальности")
        return g

    def test_field_kx_right_view_from_zero(self) -> AssertionGroup:
        """Поле = вид С НУЛЕВОЙ дальности (az=π): +kx → ВПРАВО (больше x), как видит наблюдатель РЛС.

        3D-куб — облётный ракурс с ДРУГОЙ стороны, там +kx влево. Это НЕ зеркальный баг:
        окна намеренно смотрят с противоположных концов дальности (правка Alex 2026-07-19).
        """
        g = AssertionGroup("camera.field_kx_right")
        g.add(self.field.project(20, 0, 0)[0] > self.field.project(-20, 0, 0)[0],
              "поле: +kx вправо (вид с нуля дальности)")
        g.add(self.scene.project(20, 0, 2000)[0] < self.scene.project(-20, 0, 2000)[0],
              "3D-куб: +kx влево (облётный ракурс с другой стороны)")
        return g

    def test_ky_up_both_windows(self) -> AssertionGroup:
        """+ky → вверх в ОБОИХ окнах (разворот по дальности НЕ трогает вертикаль)."""
        g = AssertionGroup("camera.ky_up")
        g.add(self.field.project(0, 20, 0)[1] < self.field.project(0, -20, 0)[1], "поле: ky вверх")
        g.add(self.scene.project(0, 20, 2000)[1] < self.scene.project(0, -20, 2000)[1],
              "3D: ky вверх")
        return g

    def test_field_horizontal_flip_vs_scene(self) -> AssertionGroup:
        """Поле развёрнуто по дальности относительно 3D: горизонталь kx ЗЕРКАЛЬНА, вертикаль ky — НЕТ.

        Наблюдатель смотрит из r=0 вдоль +r (поле), 3D облетает с макс. дальности ⇒ лево-право kx
        противоположны by construction, а угол места ky совпадает (общий метод project, разные az)."""
        g = AssertionGroup("camera.field_flip")
        for kx in (-30, -10, 5, 25):
            opposite = (self.field.project(kx, 0, 0)[0] < 0) != (self.scene.project(kx, 0, 2000)[0] < 0)
            g.add(opposite or kx == 0, f"kx={kx}: сторона X поля и 3D противоположна (вид с двух концов r)")
        for ky in (-30, -10, 5, 25):
            same = (self.field.project(0, ky, 0)[1] < 0) == (self.scene.project(0, ky, 2000)[1] < 0)
            g.add(same or ky == 0, f"ky={ky}: сторона Y (угол места) поля и 3D совпадает")
        return g

    def test_zero_range_near_and_low(self) -> AssertionGroup:
        """r=0 ближе (меньше depth) и ниже (больше y) ⇒ приближение = движение на зрителя вниз."""
        g = AssertionGroup("camera.zero_range")
        g.add(self.scene.project(0, 0, 0)[2] < self.scene.project(0, 0, 4096)[2],
              "r=0 должно быть ближе r=max (меньше depth)")
        g.add(self.scene.project(0, 0, 0)[1] > self.scene.project(0, 0, 4096)[1],
              "r=0 ниже (больше y): приближение = вниз на зрителя")
        return g

    def test_full_rotation(self) -> AssertionGroup:
        """Камера вращается ПОЛНОСТЬЮ: ракурс меняется; облёт на 180° честно меняет лево-право."""
        g = AssertionGroup("camera.full_rotation")
        turned = self.scene.rotated(self.scene.az + 1.0, self.scene.el).project(20, 10, 1000)
        base = self.scene.project(20, 10, 1000)
        g.add(abs(base[0]-turned[0]) > 1e-3 or abs(base[1]-turned[1]) > 1e-3,
              "поворот должен менять проекцию (вращение не заблокировано)")
        back = self.scene.rotated(self.scene.az + math.pi, self.scene.el)
        g.add((back.project(25, 0, 1000)[0] < back.project(-25, 0, 1000)[0]) !=
              (self.scene.project(25, 0, 1000)[0] < self.scene.project(-25, 0, 1000)[0]),
              "облёт на 180° меняет лево-право (обошли вокруг — физически верно)")
        return g


if __name__ == "__main__":
    sys.exit(0 if CameraTests().run_all() else 1)
