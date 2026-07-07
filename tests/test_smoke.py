"""Smoke-тест: сквозной прогон сцены + детерминированная классификация.

Запуск:  python tests/test_smoke.py
"""
from __future__ import annotations

from common.runner import AssertionGroup, SkipTest, TestRunner


class SmokeTests(TestRunner):

    def setup(self) -> None:
        from core.config import default_scenario
        from core.models import AxisWindows, Fft3DModel, HannWindow, RuleBasedClassifier

        self.cfg = default_scenario()
        self.model = Fft3DModel(
            self.cfg.array, self.cfg.range,
            windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()),
        )
        self.clf = RuleBasedClassifier()

    def test_imports(self) -> AssertionGroup:
        g = AssertionGroup("smoke.imports")
        g.add(self.model is not None, "Fft3DModel создан")
        g.add(self.clf is not None, "RuleBasedClassifier создан")
        return g

    def test_pipeline_and_classify(self) -> AssertionGroup:
        from core.controller import SimulationController
        from core.data_context import DataContext

        g = AssertionGroup("smoke.pipeline")
        ctrl = SimulationController(model=self.model,
                                    data_context=DataContext(root="./out/data"))
        outcome = ctrl.run(self.cfg, save_as="smoke_scene")
        cube = outcome.spectral_cube
        g.add(cube is not None, "получен SpectralCube")

        label = self.clf.classify(cube)
        g.add(label is not None, "классификатор вернул метку")
        return g

    def test_torch_optional(self) -> AssertionGroup:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise SkipTest("torch не установлен — CNN-путь пропущен") from exc
        g = AssertionGroup("smoke.torch")
        g.add(True, "torch доступен")
        return g


if __name__ == "__main__":
    SmokeTests().run_all()
