"""Тесты трекинга детекций между тактами (гл.4-бис §4.4, гл.5 §5.7 -- "летит не из
куба", группу ведут трекингом от такта к такту).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_tracking.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.models.tokenizer.arbiter import TargetDecision  # noqa: E402
from core.models.tracking import NearestNeighborTracker  # noqa: E402


def _decision(kx: float, ky: float, lead_r: int, decision: str = "target") -> TargetDecision:
    return TargetDecision(kx=kx, ky=ky, lead_r=lead_r, decision=decision, reason="single")


class NearestNeighborTrackerTests(TestRunner):

    def setup(self) -> None:
        self.tracker = NearestNeighborTracker()

    def test_moving_target_single_track_with_range_velocity(self) -> AssertionGroup:
        """§4-бис.4: движущаяся цель (растущая/убывающая дальность + угловой дрейф)
        -- ОДИН трек, is_moving=True, vel_r ~= заданной скорости сближения."""
        g = AssertionGroup("tracking.moving_target")
        leads = [100, 90, 80, 70]
        kxs = [0.0, 1.0, 2.0, 3.0]
        tracks = None
        for tact, (r, kx) in enumerate(zip(leads, kxs)):
            tracks = self.tracker.update([_decision(kx=kx, ky=0.0, lead_r=r)], tact)

        g.add(tracks is not None and len(tracks) == 1,
              f"должен быть 1 активный трек, получено {len(tracks) if tracks else 0}")
        if tracks:
            t = tracks[0]
            g.add(len(t.history) == 4, f"history должна содержать 4 такта, получено {len(t.history)}")
            g.add(abs(t.vel_r - (-10.0)) < 1e-6, f"vel_r должен быть ~=-10.0, получено {t.vel_r}")
            g.add(t.is_moving, "движущаяся цель -- is_moving должен быть True")
            g.add(t.lead_r == 70, f"последняя дальность должна быть 70, получено {t.lead_r}")
        return g

    def test_static_target_track_not_moving(self) -> AssertionGroup:
        """Статичная цель (тот же угол/дальность по тактам) -- трек, is_moving=False, vel_r~=0."""
        g = AssertionGroup("tracking.static_target")
        tracks = None
        for tact in range(4):
            tracks = self.tracker.update([_decision(kx=5.0, ky=-2.0, lead_r=50)], tact)

        g.add(tracks is not None and len(tracks) == 1,
              f"должен быть 1 активный трек, получено {len(tracks) if tracks else 0}")
        if tracks:
            t = tracks[0]
            g.add(abs(t.vel_r) < 1e-9, f"vel_r должен быть ~=0, получено {t.vel_r}")
            g.add(not t.is_moving, "статичная цель -- is_moving должен быть False")
        return g

    def test_two_targets_different_angles_two_tracks(self) -> AssertionGroup:
        """Две цели под разными (далеко разнесёнными) углами -- два разных трека,
        не перепутаны между тактами."""
        g = AssertionGroup("tracking.two_targets")
        tracker = NearestNeighborTracker()
        tracks = None
        for tact in range(3):
            decisions = [
                _decision(kx=0.0, ky=0.0, lead_r=50),
                _decision(kx=8.0, ky=8.0, lead_r=60),
            ]
            tracks = tracker.update(decisions, tact)

        g.add(tracks is not None and len(tracks) == 2,
              f"должно быть 2 активных трека, получено {len(tracks) if tracks else 0}")
        if tracks and len(tracks) == 2:
            ids = {t.track_id for t in tracks}
            g.add(len(ids) == 2, "track_id должны быть разными (треки не перепутаны)")
            by_angle = {(round(t.kx), round(t.ky)): t for t in tracks}
            g.add((0, 0) in by_angle and (8, 8) in by_angle,
                  "оба угла должны сохраниться на своих треках")
            if (0, 0) in by_angle:
                g.add(len(by_angle[(0, 0)].history) == 3, "трек (0,0) должен жить все 3 такта")
            if (8, 8) in by_angle:
                g.add(len(by_angle[(8, 8)].history) == 3, "трек (8,8) должен жить все 3 такта")
        return g

    def test_track_dies_after_max_missed_tacts(self) -> AssertionGroup:
        """Пропажа: цель исчезает -- трек живёт max_missed тактов без ассоциации, потом умирает."""
        g = AssertionGroup("tracking.track_dies")
        tracker = NearestNeighborTracker(max_missed=2)
        track_id = None
        tracks = tracker.update([_decision(kx=1.0, ky=1.0, lead_r=30)], 0)
        g.add(len(tracks) == 1, "рождение трека на такте 0")
        if tracks:
            track_id = tracks[0].track_id

        tracks = tracker.update([], 1)  # missed=1
        g.add(len(tracks) == 1 and tracks[0].track_id == track_id,
              f"missed=1 (<=max_missed=2) -- трек ещё жив, получено {len(tracks)}")
        g.add(tracks[0].missed == 1 if tracks else False, "missed должен быть 1")

        tracks = tracker.update([], 2)  # missed=2
        g.add(len(tracks) == 1, f"missed=2 (<=max_missed=2) -- трек ещё жив, получено {len(tracks)}")

        tracks = tracker.update([], 3)  # missed=3 > max_missed=2 -- смерть
        g.add(len(tracks) == 0, f"missed=3 (>max_missed=2) -- трек должен умереть, получено {len(tracks)}")
        return g

    def test_track_born_in_late_tact(self) -> AssertionGroup:
        """Рождение: новая цель появляется в позднем такте -- новый трек, не путается со старым."""
        g = AssertionGroup("tracking.track_born_late")
        tracker = NearestNeighborTracker()
        tracker.update([_decision(kx=0.0, ky=0.0, lead_r=20)], 0)
        tracker.update([_decision(kx=0.0, ky=0.0, lead_r=20)], 1)
        # такт 2: старая цель + новая, далеко по углу
        tracks = tracker.update(
            [_decision(kx=0.0, ky=0.0, lead_r=20), _decision(kx=9.0, ky=-9.0, lead_r=15)], 2,
        )
        g.add(len(tracks) == 2, f"должно быть 2 трека (старый + новорождённый), получено {len(tracks)}")
        if len(tracks) == 2:
            new_tracks = [t for t in tracks if round(t.kx) == 9]
            g.add(len(new_tracks) == 1, "новый трек должен присутствовать под своим углом")
            if new_tracks:
                g.add(len(new_tracks[0].history) == 1,
                      f"новый трек должен иметь 1 такт в history, получено {len(new_tracks[0].history)}")
        return g

    def test_false_and_jammer_decisions_are_not_tracked(self) -> AssertionGroup:
        """Ложь/помеха (`decision != 'target'`) в трекинг не берутся -- не порождают треки."""
        g = AssertionGroup("tracking.false_jammer_not_tracked")
        tracker = NearestNeighborTracker()
        decisions = [
            _decision(kx=0.0, ky=0.0, lead_r=10, decision="false"),
            _decision(kx=5.0, ky=5.0, lead_r=20, decision="jammer"),
        ]
        tracks = tracker.update(decisions, 0)
        g.add(len(tracks) == 0, f"false/jammer не должны рождать треки, получено {len(tracks)}")
        return g

    def test_update_does_not_mutate_input_decisions(self) -> AssertionGroup:
        g = AssertionGroup("tracking.does_not_mutate_input")
        decisions = [_decision(kx=1.0, ky=1.0, lead_r=30)]
        snapshot = list(decisions)
        _ = self.tracker.update(decisions, 0)
        g.add(decisions == snapshot, "update() не должен мутировать входной список decisions")
        return g


if __name__ == "__main__":
    ok = NearestNeighborTrackerTests().run_all()
    sys.exit(0 if ok else 1)
