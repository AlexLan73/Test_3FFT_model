"""Интеграционный тест полного конвейера базы (end-to-end, multi-такт).

Проверяет, что все слои стыкуются в один прогон по тактам:
  cube → VolumeTokenizer → assemble_range → EdgeArbiter → BeamTargeting  (CognitiveCycle, §8.3)
                                                        → NearestNeighborTracker (§4-бис.4, между тактами)

Движущаяся цель (пик сдвигается по дальности от такта к такту) → один устойчивый трек «летит».
🚫 pytest -- только TestRunner (правило 04). Запуск:  python tests/test_integration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.models.result import Axis, SpectralCube  # noqa: E402
from core.models.targeting import BeamTargeting, CognitiveCycle  # noqa: E402
from core.models.tokenizer import EdgeArbiter, VolumeTokenizer  # noqa: E402
from core.models.tracking import NearestNeighborTracker  # noqa: E402


def _cube_with_target(ix: int, iy: int, r0: int, seed: int, nx: int = 16, ny: int = 16,
                      n: int = 24) -> SpectralCube:
    """Куб с компактной целью на угле (ix,iy), пик в бинах r0..r0+2 (протяжённость отклика)."""
    rng = np.random.default_rng(seed)
    mag = np.abs(rng.standard_normal((nx, ny, n)) + 1j * rng.standard_normal((nx, ny, n))) * 0.1
    for r in range(r0, min(r0 + 3, n)):
        mag[ix, iy, r] = 8.0
    kx = Axis("kx", np.arange(-nx // 2, nx // 2), centered=True)
    ky = Axis("ky", np.arange(-ny // 2, ny // 2), centered=True)
    rng_ax = Axis("range", np.arange(n) * 10.0, centered=False)
    return SpectralCube(mag, kx, ky, rng_ax)


class FullPipelineIntegrationTests(TestRunner):
    """End-to-end: CognitiveCycle (гейт→арбитр→целеуказание) + трекинг между тактами."""

    def test_moving_target_end_to_end(self) -> AssertionGroup:
        g = AssertionGroup("integration.moving_target")
        cycle = CognitiveCycle(VolumeTokenizer(window_l=1), EdgeArbiter(),
                               BeamTargeting(cone_half=1, max_beams=7))
        tracker = NearestNeighborTracker(gate=50.0, w_r=0.1, max_missed=2, moving_threshold=0.5)

        # цель на угле (ix=9,iy=7), пик движется по дальности: r0 = 8,9,10,11,12
        ix, iy = 9, 7
        tracks_final: list = []
        beams_seen = 0
        for tact, r0 in enumerate((8, 9, 10, 11, 12)):
            cube = _cube_with_target(ix, iy, r0, seed=tact)
            res = cycle.step(cube)                       # tokenize→arbitrate→target
            beams_seen += len(res.beams)
            tracks_final = tracker.update(list(res.decisions), tact)   # трекинг между тактами

        g.add(beams_seen > 0, "целеуказание сформировало пучки хотя бы на части тактов")
        # трек цели существует и «летит» (пик двигался по дальности)
        moving = [t for t in tracks_final if t.is_moving]
        g.add(len(tracks_final) >= 1, f"после 5 тактов есть активные треки, получено {len(tracks_final)}")
        g.add(any(t.age >= 2 for t in tracks_final),
              "хотя бы один трек устойчив (age>=2) -- цель прослежена между тактами")
        g.add(len(moving) >= 1, "движущаяся цель помечена is_moving (§4-бис.4 «летит» из треков)")
        return g

    def test_pipeline_layers_consistent(self) -> AssertionGroup:
        """decisions арбитра согласованы с beams целеуказания (по одной команде на target)."""
        g = AssertionGroup("integration.layers_consistent")
        cycle = CognitiveCycle(VolumeTokenizer(window_l=1), EdgeArbiter(), BeamTargeting())
        res = cycle.step(_cube_with_target(9, 7, 10, seed=0))
        n_targets = sum(1 for d in res.decisions if d.decision == "target")
        g.add(len(res.beams) == n_targets,
              f"пучков={len(res.beams)} == target-решений={n_targets} (слои стыкуются)")
        g.add(len(res.verdicts) >= len(res.decisions),
              "вердиктов прохода-2 не меньше, чем решений арбитра")
        return g


if __name__ == "__main__":
    ok = FullPipelineIntegrationTests().run_all()
    sys.exit(0 if ok else 1)
