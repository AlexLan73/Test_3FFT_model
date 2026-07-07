"""Тесты CA-CFAR детектора (БЕЗ pytest — только TestRunner).

Запуск:  python tests/test_cfar.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import ArrayConfig, BarrageSpec, RangeConfig, SceneConfig, TargetSpec  # noqa: E402
from core.config.scene_config import ThermalNoiseSpec  # noqa: E402
from core.generators.scene import SceneBuilder, Synthesizer  # noqa: E402
from core.models import AxisWindows, Fft3DModel, HannWindow  # noqa: E402
from core.models.anti_barrage import CaCfarDetector, SubspaceNuller  # noqa: E402

# ── Общие параметры сцены ─────────────────────────────────────────────────────

_ARRAY = ArrayConfig(16, 16)
_RNG   = RangeConfig(n_real=16, n_fft=64)
_TARGET_KX, _TARGET_KY, _TARGET_RANGE = 2.0, 0.0, 8.0


def _model() -> Fft3DModel:
    return Fft3DModel(_ARRAY, _RNG,
                      windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()))


def _cube_from(emitters: tuple, seed: int = 1, null: bool = False):
    cfg = SceneConfig(emitters=emitters, thermal=ThermalNoiseSpec(power=0.02))
    raw = Synthesizer(_ARRAY, _RNG, seed=seed).build(SceneBuilder().build(cfg))
    if null:
        raw = SubspaceNuller(n_jammers=1).apply(raw)
    return _model().process(raw)


# ── Тест-набор ─────────────────────────────────────────────────────────────────

class CfarTests(TestRunner):

    def setup(self) -> None:
        self.cfar = CaCfarDetector(pfa=1e-3, n_train=8, n_guard=4)

    # ── 1. Формула порога α = N(P_fa^(−1/N) − 1) ──────────────────────────────
    def test_alpha_formula(self) -> AssertionGroup:
        g = AssertionGroup("cfar.alpha_formula")
        n = 2 * self.cfar.n_train
        expected = n * (self.cfar.pfa ** (-1.0 / n) - 1.0)
        g.add(abs(self.cfar.alpha - expected) < 1e-12,
              f"alpha={self.cfar.alpha:.6f} должно = {expected:.6f}")
        g.add(self.cfar.alpha > 0.0, "alpha должно быть > 0")
        return g

    # ── 2. Цель детектируется в правильном бине дальности ──────────────────────
    def test_target_detected(self) -> AssertionGroup:
        g = AssertionGroup("cfar.target_detected")
        cube = _cube_from((TargetSpec(kx=_TARGET_KX, ky=_TARGET_KY,
                                      range_bin=_TARGET_RANGE, amplitude=1.0),))
        ix, iy = cube.index_of_angle(_TARGET_KX, _TARGET_KY)
        dets = self.cfar.detect_cell(cube, ix, iy)
        hit = any(abs(d.range_bin - int(_TARGET_RANGE)) <= 1 for d in dets)
        g.add(hit, f"цель ожидается в бине ~{int(_TARGET_RANGE)}, "
                   f"детекции={[d.range_bin for d in dets]}")
        return g

    # ── 3. Монотонность: меньший P_fa → не больше детекций ────────────────────
    def test_pfa_monotonic(self) -> AssertionGroup:
        g = AssertionGroup("cfar.pfa_monotonic")
        cube = _cube_from((TargetSpec(kx=_TARGET_KX, ky=_TARGET_KY,
                                      range_bin=_TARGET_RANGE, amplitude=1.0),))
        loose = CaCfarDetector(pfa=1e-2, n_train=8, n_guard=4)
        tight = CaCfarDetector(pfa=1e-5, n_train=8, n_guard=4)
        n_loose = len(loose.detect(cube))
        n_tight = len(tight.detect(cube))
        g.add(tight.alpha > loose.alpha,
              f"меньший P_fa → больший alpha ({tight.alpha:.2f} > {loose.alpha:.2f})")
        g.add(n_tight <= n_loose,
              f"меньший P_fa → не больше детекций ({n_tight} <= {n_loose})")
        return g

    # ── 4. Чистый шум: мало ложных тревог ─────────────────────────────────────
    def test_noise_false_alarms(self) -> AssertionGroup:
        g = AssertionGroup("cfar.noise_false_alarms")
        cube = _cube_from(())  # только тепловой шум, без эмиттеров
        dets = self.cfar.detect(cube)
        nx, ny, nr = cube.magnitude.shape
        n_cells = nx * ny * nr
        # ожидаем ~ P_fa · N_ячеек; допускаем щедрый запас x50 (край/локмакс/окно)
        limit = max(10, int(self.cfar.pfa * n_cells * 50))
        g.add(len(dets) <= limit,
              f"ложных тревог {len(dets)} должно быть <= {limit} (P_fa={self.cfar.pfa})")
        return g

    # ── 5. Full-chain: barrage+target → нуллер → CFAR → цель есть ──────────────
    def test_full_chain(self) -> AssertionGroup:
        g = AssertionGroup("cfar.full_chain")
        emitters = (
            TargetSpec(kx=_TARGET_KX, ky=_TARGET_KY, range_bin=_TARGET_RANGE, amplitude=1.0),
            BarrageSpec(kx=-4.0, ky=0.0, power=6.0),
        )
        cube = _cube_from(emitters, null=True)
        ixt, iyt = cube.index_of_angle(_TARGET_KX, _TARGET_KY)
        dets = self.cfar.detect(cube)
        target_hit = any(d.kx_idx == ixt and d.ky_idx == iyt
                         and abs(d.range_bin - int(_TARGET_RANGE)) <= 1 for d in dets)
        g.add(target_hit, "цель должна детектироваться после нуллинга в своей ячейке")
        # сильнейшая детекция — на дальности цели (пик поля)
        if dets:
            top = max(dets, key=lambda d: d.level_db)
            g.add(abs(top.range_bin - int(_TARGET_RANGE)) <= 1,
                  f"сильнейшая детекция r={top.range_bin} должна быть у цели ~{int(_TARGET_RANGE)}")
        return g

    # ── 6. detect() не мутирует куб ───────────────────────────────────────────
    def test_cube_immutability(self) -> AssertionGroup:
        g = AssertionGroup("cfar.cube_immutability")
        cube = _cube_from((TargetSpec(kx=_TARGET_KX, ky=_TARGET_KY,
                                      range_bin=_TARGET_RANGE, amplitude=1.0),))
        before = cube.magnitude.copy()
        _ = self.cfar.detect(cube)
        diff = float(np.max(np.abs(cube.magnitude - before)))
        g.add(diff == 0.0, f"detect() не должен мутировать куб: max diff={diff}")
        return g


if __name__ == "__main__":
    CfarTests().run_all()
