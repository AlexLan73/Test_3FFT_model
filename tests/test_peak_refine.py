"""Тесты параболического уточнения пика (🚫 pytest, правило 04).

Запуск:  .venv/Scripts/python.exe tests/test_peak_refine.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работала форма `python tests/test_peak_refine.py` (конвенция репо).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import ArrayConfig, ProjectConfig  # noqa: E402
from core.generators.grid import ArrayGrid  # noqa: E402
from core.generators.waveforms import AmToCube  # noqa: E402
from core.models.tokenizer import axis_value_at, parabolic_delta, refine_peak  # noqa: E402


def _gaussian_1d(n: int, center: float, sigma: float = 1.2) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return np.exp(-((k - center) ** 2) / (2.0 * sigma ** 2))


class PeakRefineTests(TestRunner):

    def test_delta_gaussian_exact(self) -> AssertionGroup:
        """Лог-парабола ТОЧНА на гауссовом пике: δ восстанавливается до машинной точности."""
        g = AssertionGroup("peak_refine.delta_gaussian")
        for frac in (-0.4, -0.15, 0.0, 0.27, 0.49):
            p = _gaussian_1d(16, 7.0 + frac)
            d = parabolic_delta(p[6], p[7], p[8])
            g.add(abs(d - frac) < 1e-9, f"гаусс frac={frac}: δ={d:.6f}, ошибка > 1e-9")
        return g

    def test_delta_degenerate(self) -> AssertionGroup:
        """Не-пик/плоскость/нулевая мощность — честный 0, диапазон всегда [−½, ½]."""
        g = AssertionGroup("peak_refine.degenerate")
        g.add(parabolic_delta(1.0, 1.0, 1.0) == 0.0, "плоскость должна давать δ=0")
        g.add(parabolic_delta(2.0, 1.0, 2.0) == 0.0, "провал (не пик) должен давать δ=0")
        g.add(parabolic_delta(0.0, 0.0, 0.0) == 0.0, "нулевая мощность не должна ронять log")
        d = parabolic_delta(1e-12, 1.0, 0.999999999)
        g.add(-0.5 <= d <= 0.5, f"δ обязан лежать в [−½,½], получено {d}")
        return g

    def test_refine_3d_gaussian(self) -> AssertionGroup:
        """3D-пик с дробным центром по всем осям: frac_index сходится к истине."""
        g = AssertionGroup("peak_refine.refine_3d")
        true_c = (7.3, 4.6, 19.8)
        px = _gaussian_1d(16, true_c[0])
        py = _gaussian_1d(12, true_c[1])
        pz = _gaussian_1d(32, true_c[2])
        cube = px[:, None, None] * py[None, :, None] * pz[None, None, :]

        before = cube.copy()
        rp = refine_peak(cube)
        g.add(np.array_equal(cube, before), "refine_peak не должен мутировать вход")
        g.add(rp.index == (7, 5, 20), f"целый argmax должен быть (7,5,20), получено {rp.index}")
        for ax, (f, t) in enumerate(zip(rp.frac_index, true_c, strict=True)):
            g.add(abs(f - t) < 1e-6, f"ось {ax}: frac={f:.4f}, истина {t}")
        return g

    def test_edge_no_refine(self) -> AssertionGroup:
        """Пик на краю оси: соседа нет — δ=0 по этой оси (без заворота)."""
        g = AssertionGroup("peak_refine.edge")
        p = np.zeros((8, 8))
        p[0, 3] = 1.0
        p[0, 2] = p[0, 4] = 0.5
        rp = refine_peak(p)
        g.add(rp.index == (0, 3), f"argmax должен быть (0,3), получено {rp.index}")
        g.add(rp.delta[0] == 0.0, "на краю оси 0 поправка обязана быть 0")
        g.add(rp.delta[1] == 0.0, "симметричные соседи ⇒ δ=0 по оси 1")
        return g

    def test_explicit_index(self) -> AssertionGroup:
        """Уточнение ЗАДАННОГО пика (не глобального argmax) — путь для CFAR/грубой карты."""
        g = AssertionGroup("peak_refine.explicit_index")
        p = _gaussian_1d(32, 10.3) + 2.0 * _gaussian_1d(32, 24.0)
        rp = refine_peak(p, index=(10,))
        g.add(abs(rp.frac_index[0] - 10.3) < 0.05,
              f"слабый пик 10.3: frac={rp.frac_index[0]:.3f}")
        bad = False
        try:
            refine_peak(p, index=(1, 2))
        except ValueError:
            bad = True
        g.add(bad, "index неверной размерности обязан давать ValueError")
        return g

    def test_axis_value_at(self) -> AssertionGroup:
        """Дробный индекс → физическое значение центрированной угловой оси."""
        g = AssertionGroup("peak_refine.axis_value")
        values = np.arange(-8, 8)                     # kx-ось куба 16
        g.add(axis_value_at(values, 7.5) == -0.5, "индекс 7.5 на kx-оси 16 должен дать −0.5")
        g.add(axis_value_at(values, 0.0) == -8.0, "индекс 0 должен дать первый бин")
        return g

    def test_cube_fractional_steering(self) -> AssertionGroup:
        """Интеграция с трактом: steering на ДРОБНЫЙ (kx,ky) + дробный частотный бин →
        куб AmToCube → парабола бьёт argmax по всем 3 осям."""
        g = AssertionGroup("peak_refine.cube_integration")
        nx = ny = 16
        depth = 32
        true_kx, true_ky = -4.6, 2.3                  # дробные угловые бины
        true_iz = 6.4                                  # дробный частотный бин (f=fs·6.4/32)

        n = np.arange(depth)
        tone = np.exp(2j * np.pi * true_iz / depth * n)
        steer = ArrayGrid(nx, ny).steering(true_kx, true_ky)
        volume = (steer[:, :, None] * tone[None, None, :]).astype(np.complex64)

        cfg = ProjectConfig(array=ArrayConfig(nx, ny), modulation="am")
        cube = AmToCube(depth=depth, step=depth // 2).fill(volume, cfg)
        power = cube.magnitude.astype(np.float64) ** 2

        rp = refine_peak(power)
        truth = (true_kx + nx / 2, true_ky + ny / 2, true_iz)
        for ax, (f, t) in enumerate(zip(rp.frac_index, truth, strict=True)):
            err_ref = abs(f - t)
            err_arg = abs(rp.index[ax] - t)
            g.add(err_ref < 0.15, f"ось {ax}: ошибка параболы {err_ref:.3f} ≥ 0.15 бина")
            g.add(err_ref <= err_arg + 1e-9,
                  f"ось {ax}: парабола ({err_ref:.3f}) не должна проигрывать argmax ({err_arg:.3f})")
        kx_hat = axis_value_at(cube.kx.values, rp.frac_index[0])
        g.add(abs(kx_hat - true_kx) < 0.15, f"kx физ.: {kx_hat:.3f} vs истина {true_kx}")
        return g


if __name__ == "__main__":
    sys.exit(0 if PeakRefineTests().run_all() else 1)
