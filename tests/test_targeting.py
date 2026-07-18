"""Тесты целеуказания пучка FM-m (гл.8, `Doc/Patent/glava8_celeukazanie.md`).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_targeting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.models.targeting import (  # noqa: E402
    BeamCommand,
    BeamTargeting,
    CognitiveCycle,
    CycleResult,
)
from core.models.tokenizer import EdgeArbiter, VolumeTokenizer  # noqa: E402
from core.models.tokenizer.arbiter import TargetDecision  # noqa: E402


def _target(kx: float, ky: float, lead_r: int) -> TargetDecision:
    return TargetDecision(kx=kx, ky=ky, lead_r=lead_r, decision="target", reason="single")


def _jammer(kx: float, ky: float) -> TargetDecision:
    return TargetDecision(kx=kx, ky=ky, lead_r=0, decision="jammer", reason="barrage_selfscreen")


class BeamTargetingTests(TestRunner):
    """§8.2: решения арбитра (target) -> пучок лучей в конус неопределённости."""

    def test_target_yields_command_with_center(self) -> AssertionGroup:
        g = AssertionGroup("targeting.target_command")
        cmds = BeamTargeting(cone_half=1, max_beams=7).point([_target(2.0, 3.0, 42)])
        g.add(len(cmds) == 1, f"1 цель -> 1 команда, получено {len(cmds)}")
        c = cmds[0]
        g.add(c.target_r == 42, "target_r = lead_r цели")
        g.add(c.center_kx == 2.0 and c.center_ky == 3.0, "центр = угол цели")
        g.add((2.0, 3.0) in c.beam_angles, "пучок включает сам центр")
        g.add(c.beam_angles[0] == (2.0, 3.0), "центр -- первый луч (по возрастанию расстояния)")
        return g

    def test_jammer_and_false_yield_no_command(self) -> AssertionGroup:
        g = AssertionGroup("targeting.jammer_no_command")
        decisions = [
            _target(0.0, 0.0, 10),
            _jammer(1.0, 1.0),
            TargetDecision(kx=2.0, ky=2.0, lead_r=5, decision="false", reason="stale_code"),
        ]
        cmds = BeamTargeting().point(decisions)
        g.add(len(cmds) == 1, f"только по цели светим (§8), получено команд {len(cmds)}")
        g.add(cmds[0].center_kx == 0.0, "команда -- у цели, не у jammer/false")
        return g

    def test_two_targets_two_commands(self) -> AssertionGroup:
        g = AssertionGroup("targeting.two_targets")
        cmds = BeamTargeting().point([_target(2.0, 3.0, 10), _target(-4.0, 1.0, 20)])
        g.add(len(cmds) == 2, "2 цели -> 2 команды")
        centers = {(c.center_kx, c.center_ky) for c in cmds}
        g.add(centers == {(2.0, 3.0), (-4.0, 1.0)}, "центры соответствуют целям")
        return g

    def test_cone_covers_neighbors(self) -> AssertionGroup:
        g = AssertionGroup("targeting.cone_neighbors")
        # cone_half=1, max_beams=9 -> покрыть весь 3x3 конус вокруг (0,0)
        cmds = BeamTargeting(cone_half=1, max_beams=9, step=1.0).point([_target(0.0, 0.0, 5)])
        angles = set(cmds[0].beam_angles)
        expected = {(float(dx), float(dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        g.add(angles == expected, f"конус ±1 покрыт (9 лучей), получено {len(angles)}")
        return g

    def test_max_beams_limits(self) -> AssertionGroup:
        g = AssertionGroup("targeting.max_beams")
        cmds = BeamTargeting(cone_half=2, max_beams=5).point([_target(0.0, 0.0, 5)])
        g.add(len(cmds[0].beam_angles) == 5, "max_beams ограничивает число лучей до 5")
        g.add(cmds[0].beam_angles[0] == (0.0, 0.0), "ближайший (центр) остаётся первым")
        return g

    def test_validation(self) -> AssertionGroup:
        g = AssertionGroup("targeting.validation")
        for bad in [dict(cone_half=-1), dict(max_beams=0), dict(step=0.0)]:
            raised = False
            try:
                BeamTargeting(**bad)
            except ValueError:
                raised = True
            g.add(raised, f"{bad} должен кидать ValueError")
        return g

    def test_does_not_mutate_input(self) -> AssertionGroup:
        g = AssertionGroup("targeting.no_mutation")
        decisions = [_target(1.0, 1.0, 7)]
        BeamTargeting().point(decisions)
        g.add(len(decisions) == 1 and decisions[0].kx == 1.0, "вход не мутирован")
        return g


class CognitiveCycleTests(TestRunner):
    """§8.3: петля токенизатор -> арбитр -> целеуказание (Facade над готовым)."""

    def setup(self) -> None:
        from core.models.result import Axis, SpectralCube

        nx, ny, n = 16, 16, 24
        rng = np.random.default_rng(0)
        mag = np.abs(rng.standard_normal((nx, ny, n)) + 1j * rng.standard_normal((nx, ny, n))) * 0.1
        # компактная "цель" на (ix=9,iy=7) в нескольких соседних бинах дальности
        for r in (10, 11, 12):
            mag[9, 7, r] = 8.0
        kx = Axis("kx", np.arange(-nx // 2, nx // 2), centered=True)
        ky = Axis("ky", np.arange(-ny // 2, ny // 2), centered=True)
        rng_ax = Axis("range", np.arange(n) * 10.0, centered=False)
        self.cube = SpectralCube(mag, kx, ky, rng_ax)

    def test_step_returns_consistent_result(self) -> AssertionGroup:
        g = AssertionGroup("targeting.cycle_step")
        cycle = CognitiveCycle(VolumeTokenizer(window_l=1), EdgeArbiter(),
                               BeamTargeting(cone_half=1, max_beams=7))
        res = cycle.step(self.cube)
        g.add(isinstance(res, CycleResult), "step -> CycleResult")
        # число пучков == числу target-решений (§8: пучок на цель)
        n_targets = sum(1 for d in res.decisions if d.decision == "target")
        g.add(len(res.beams) == n_targets,
              f"пучков = target-решений: beams={len(res.beams)} targets={n_targets}")
        # каждый пучок центрирован на своём target-решении
        target_angles = {(d.kx, d.ky) for d in res.decisions if d.decision == "target"}
        beam_centers = {(b.center_kx, b.center_ky) for b in res.beams}
        g.add(beam_centers <= target_angles, "центры пучков -- на target-решениях")
        return g

    def test_step_does_not_mutate_cube(self) -> AssertionGroup:
        g = AssertionGroup("targeting.cycle_no_mutation")
        before = self.cube.magnitude.copy()
        CognitiveCycle(VolumeTokenizer(window_l=1), EdgeArbiter(), BeamTargeting()).step(self.cube)
        g.add(bool(np.array_equal(before, self.cube.magnitude)), "куб не мутирован петлёй")
        return g


if __name__ == "__main__":
    ok = True
    for cls in (BeamTargetingTests, CognitiveCycleTests):
        ok = cls().run_all() and ok
    sys.exit(0 if ok else 1)
