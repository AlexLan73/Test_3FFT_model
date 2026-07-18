"""Тесты ROI-гейта детекций (гл.8 целеуказание -> гейт CFAR, `roi_gate.py`).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_roi_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.models.anti_barrage.cfar import Detection  # noqa: E402
from core.models.targeting import BeamCommand, RoiGate  # noqa: E402


def _det(kx: float, ky: float, range_bin: int) -> Detection:
    return Detection(
        kx_idx=0, ky_idx=0, range_bin=range_bin,
        level_db=0.0, threshold_db=-3.0, kx=kx, ky=ky,
    )


def _beam(target_r: int, center_kx: float, center_ky: float) -> BeamCommand:
    return BeamCommand(target_r=target_r, center_kx=center_kx, center_ky=center_ky,
                       beam_angles=((center_kx, center_ky),))


class RoiGateTests(TestRunner):

    def setup(self) -> None:
        self.gate = RoiGate(angle_half=1.0, range_half=2)

    # ── 1. Детекция внутри ROI проходит ──────────────────────────────────────
    def test_detection_inside_roi_passes(self) -> AssertionGroup:
        g = AssertionGroup("roi_gate.inside_passes")
        beam = _beam(target_r=10, center_kx=2.0, center_ky=0.0)
        det = _det(kx=2.0, ky=0.0, range_bin=11)  # dr=1<=2, dkx=0, dky=0
        out = self.gate.filter([det], [beam])
        g.add(len(out) == 1 and out[0] is det, "детекция в ROI должна пройти")
        return g

    # ── 2. Детекция вне ROI (далеко по дальности) отфильтрована ──────────────
    def test_detection_far_range_filtered(self) -> AssertionGroup:
        g = AssertionGroup("roi_gate.far_range_filtered")
        beam = _beam(target_r=10, center_kx=2.0, center_ky=0.0)
        det = _det(kx=2.0, ky=0.0, range_bin=20)  # dr=10 > range_half=2
        out = self.gate.filter([det], [beam])
        g.add(len(out) == 0, "детекция далеко по дальности должна быть отфильтрована")
        return g

    # ── 3. Детекция вне ROI (далеко по углу) отфильтрована ────────────────────
    def test_detection_far_angle_filtered(self) -> AssertionGroup:
        g = AssertionGroup("roi_gate.far_angle_filtered")
        beam = _beam(target_r=10, center_kx=2.0, center_ky=0.0)
        det = _det(kx=6.0, ky=0.0, range_bin=10)  # dkx=4 > angle_half=1
        out = self.gate.filter([det], [beam])
        g.add(len(out) == 0, "детекция далеко по углу должна быть отфильтрована")
        return g

    # ── 4. Smoke: 5 детекций (2 в ROI, 3 вне) -> 1 BeamCommand -> 2 прошли ────
    def test_smoke_five_detections_two_in_roi(self) -> AssertionGroup:
        g = AssertionGroup("roi_gate.smoke_five")
        beam = _beam(target_r=10, center_kx=2.0, center_ky=0.0)
        dets = [
            _det(kx=2.0, ky=0.0, range_bin=10),    # в ROI (точно центр)
            _det(kx=1.5, ky=0.5, range_bin=11),    # в ROI (внутри окна)
            _det(kx=2.0, ky=0.0, range_bin=50),    # вне ROI -- далеко по дальности
            _det(kx=-4.0, ky=0.0, range_bin=10),   # вне ROI -- далеко по углу (заград)
            _det(kx=8.0, ky=8.0, range_bin=99),    # вне ROI -- далеко по всему
        ]
        out = self.gate.filter(dets, [beam])
        g.add(len(out) == 2, f"5 детекций (2 в ROI) -> 2 должны пройти, получено {len(out)}")
        return g

    # ── 5. Пустой beams -> пустой результат (нет целеуказания = нечего пропускать) ──
    def test_empty_beams_yields_empty(self) -> AssertionGroup:
        g = AssertionGroup("roi_gate.empty_beams")
        dets = [_det(kx=0.0, ky=0.0, range_bin=5), _det(kx=1.0, ky=1.0, range_bin=6)]
        out = self.gate.filter(dets, [])
        g.add(out == [], "без целеуказания (beams=[]) фильтр не пропускает ничего")
        return g

    # ── 6. Несколько BeamCommand -- union по ROI (проходит если попал хоть в один) ──
    def test_multiple_beams_union(self) -> AssertionGroup:
        g = AssertionGroup("roi_gate.multiple_beams_union")
        beam_a = _beam(target_r=10, center_kx=2.0, center_ky=0.0)
        beam_b = _beam(target_r=40, center_kx=-4.0, center_ky=0.0)
        dets = [
            _det(kx=2.0, ky=0.0, range_bin=10),   # в ROI beam_a
            _det(kx=-4.0, ky=0.0, range_bin=40),  # в ROI beam_b
            _det(kx=8.0, ky=8.0, range_bin=90),   # ни в одном ROI
        ]
        out = self.gate.filter(dets, [beam_a, beam_b])
        g.add(len(out) == 2, f"union ROI обоих пучков -> 2 должны пройти, получено {len(out)}")
        return g

    # ── 7. filter() не мутирует входы ────────────────────────────────────────
    def test_does_not_mutate_inputs(self) -> AssertionGroup:
        g = AssertionGroup("roi_gate.no_mutation")
        beam = _beam(target_r=10, center_kx=2.0, center_ky=0.0)
        dets = [_det(kx=2.0, ky=0.0, range_bin=10), _det(kx=9.0, ky=9.0, range_bin=90)]
        dets_before = list(dets)
        beams = [beam]
        self.gate.filter(dets, beams)
        g.add(dets == dets_before, "detections не мутированы")
        g.add(beams == [beam], "beams не мутированы")
        return g


if __name__ == "__main__":
    RoiGateTests().run_all()
