"""Тесты anti-barrage phase2: diagonal loading + AntiBarragePipeline (Facade).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_anti_barrage_pipeline.py
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
from core.models.anti_barrage import AntiBarragePipeline, CaCfarDetector, SubspaceNuller  # noqa: E402

_ARRAY = ArrayConfig(16, 16)
_RNG = RangeConfig(n_real=16, n_fft=64)
_TARGET_KX, _TARGET_KY, _TARGET_RANGE = 2.0, 0.0, 8.0


def _model() -> Fft3DModel:
    return Fft3DModel(_ARRAY, _RNG,
                      windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()))


def _raw_scene(emitters: tuple, seed: int = 1) -> np.ndarray:
    cfg = SceneConfig(emitters=emitters, thermal=ThermalNoiseSpec(power=0.02))
    return Synthesizer(_ARRAY, _RNG, seed=seed).build(SceneBuilder().build(cfg))


class DiagonalLoadingTests(TestRunner):
    """Diagonal loading: R' = R + loading·(tr(R)/M)·I -- робастность малой выборки."""

    def test_loading_reduces_condition_number(self) -> AssertionGroup:
        g = AssertionGroup("nuller.diagonal_loading_cond")
        rng = np.random.default_rng(0)
        nx, ny, k = 16, 16, 4                              # K=4 << M=256 -> R вырождена
        m = nx * ny
        x = (rng.standard_normal((m, k)) + 1j * rng.standard_normal((m, k))).astype(np.complex128)
        r = (x @ x.conj().T) / k
        cond0 = np.linalg.cond(r)
        for load in (0.01, 0.1):
            r_load = r + load * (np.trace(r).real / m) * np.eye(m)
            cond_l = np.linalg.cond(r_load)
            g.add(cond_l < cond0,
                  f"loading={load}: cond {cond_l:.3g} < без loading {cond0:.3g}")
        return g

    def test_loading_zero_is_backward_compatible(self) -> AssertionGroup:
        g = AssertionGroup("nuller.loading_zero_compat")
        raw = _raw_scene((TargetSpec(kx=_TARGET_KX, ky=_TARGET_KY,
                                     range_bin=_TARGET_RANGE, amplitude=1.0),))
        default = SubspaceNuller(n_jammers=1).apply(raw)
        load0 = SubspaceNuller(n_jammers=1, loading=0.0).apply(raw)
        g.add(bool(np.allclose(default, load0)),
              "loading=0.0 должен давать РОВНО тот же результат, что nuller без loading")
        return g

    def test_loading_affects_report_not_apply(self) -> AssertionGroup:
        """Находка (математика): diagonal loading влияет на ОЦЕНКУ с.з. (report), но НЕ на
        подавление (apply) -- проекция использует собственные ВЕКТОРЫ, а `R + λI` их не меняет
        (сдвигает лишь собственные ЗНАЧЕНИЯ). Поэтому loading полезен для детектора `is_barrage`
        (стабилизация оценки числа источников при малой K), а не для усиления подавления."""
        g = AssertionGroup("nuller.loading_report_not_apply")
        raw = _raw_scene((BarrageSpec(kx=0.0, ky=0.0, power=1.0),))
        r0 = SubspaceNuller(n_jammers=1, loading=0.0).report(raw)
        rl = SubspaceNuller(n_jammers=1, loading=0.1).report(raw)
        g.add(abs(r0.lambda_ratio - rl.lambda_ratio) > 1e-9,
              "loading ВЛИЯЕТ на report.lambda_ratio (сдвиг собственных значений)")

        a0 = SubspaceNuller(n_jammers=1, loading=0.0).apply(raw)
        al = SubspaceNuller(n_jammers=1, loading=0.1).apply(raw)
        g.add(np.all(np.isfinite(al)), "результат apply конечен")
        g.add(bool(np.allclose(a0, al)),
              "apply ИНВАРИАНТЕН к loading -- проекция на с.в., R+λI не меняет векторы (находка)")
        return g


class AntiBarragePipelineTests(TestRunner):
    """Facade §phase2: nuller -> fft_model -> cfar единым process()."""

    def setup(self) -> None:
        self.nuller = SubspaceNuller(n_jammers=1)
        self.model = _model()
        self.cfar = CaCfarDetector(pfa=1e-3, n_train=8, n_guard=4)
        self.pipe = AntiBarragePipeline(self.nuller, self.model, self.cfar)

    def test_facade_equals_manual_chain(self) -> AssertionGroup:
        g = AssertionGroup("pipeline.facade_equals_manual")
        raw = _raw_scene((TargetSpec(kx=_TARGET_KX, ky=_TARGET_KY,
                                     range_bin=_TARGET_RANGE, amplitude=1.0),))
        via_pipe = self.pipe.process(raw)
        manual = self.cfar.detect(self.model.process(self.nuller.apply(raw)))
        g.add([d.range_bin for d in via_pipe] == [d.range_bin for d in manual],
              "pipeline.process должен совпасть с ручной цепочкой nuller->fft->cfar")
        return g

    def test_target_detected_under_barrage(self) -> AssertionGroup:
        g = AssertionGroup("pipeline.target_under_barrage")
        raw = _raw_scene((
            TargetSpec(kx=_TARGET_KX, ky=_TARGET_KY, range_bin=_TARGET_RANGE, amplitude=1.0),
            BarrageSpec(kx=0.0, ky=0.0, power=5.0),          # заград с боресайта
        ))
        dets = self.pipe.process(raw)
        hit = any(abs(d.range_bin - int(_TARGET_RANGE)) <= 1 for d in dets)
        g.add(hit, f"цель ~бин {int(_TARGET_RANGE)} детектируется после подавления заграда, "
                   f"детекции={[d.range_bin for d in dets]}")
        return g

    def test_does_not_mutate_input(self) -> AssertionGroup:
        g = AssertionGroup("pipeline.no_mutation")
        raw = _raw_scene((TargetSpec(kx=_TARGET_KX, ky=_TARGET_KY,
                                     range_bin=_TARGET_RANGE, amplitude=1.0),))
        before = raw.copy()
        self.pipe.process(raw)
        g.add(bool(np.array_equal(before, raw)), "process не должен мутировать входной куб")
        return g


if __name__ == "__main__":
    ok = True
    for cls in (DiagonalLoadingTests, AntiBarragePipelineTests):
        ok = cls().run_all() and ok
    sys.exit(0 if ok else 1)
