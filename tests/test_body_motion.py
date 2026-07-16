"""Тесты P1 body-motion: ProjectConfig, MessageBus, MotionModel, Kinematics.

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_body_motion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import ArrayConfig, ProjectConfig  # noqa: E402
from core.data_context import (  # noqa: E402
    DataContext,
    MessageBus,
    project_config_from_dict,
    project_config_to_dict,
)
from core.data_context.run_workspace import from_yaml, to_yaml  # noqa: E402
from core.generators import Tact, TactSequence  # noqa: E402
from core.motion import (  # noqa: E402
    ConstantAccel,
    ConstantVelocity,
    CoordinatedTurn,
    Kinematics,
    MarkovDrift,
    TargetState,
)


class ProjectConfigTests(TestRunner):

    def test_aggregates_existing_vo(self) -> AssertionGroup:
        g = AssertionGroup("project_config.aggregates_existing_vo")
        cfg = ProjectConfig()
        g.add(isinstance(cfg.array, ArrayConfig), "array должен быть ArrayConfig")
        g.add(cfg.array.nx == 16 and cfg.array.ny == 16, "дефолт array 16x16")
        g.add(cfg.range_.n_real == 16, "дефолт range_.n_real=16")
        g.add(cfg.wave.fs == 12e6, "wave -- существующий WaveTimeConfig (fs=12e6)")
        g.add(cfg.modulation == "lfm", "дефолт modulation=lfm")
        g.add(cfg.am_step == 8, "дефолт am_step=8")
        return g

    def test_non_square_array_and_padded_shape(self) -> AssertionGroup:
        g = AssertionGroup("project_config.non_square_padded_shape")
        arr = ArrayConfig(nx=6, ny=15)
        cfg = ProjectConfig(array=arr)
        g.add(cfg.array.nx == 6 and cfg.array.ny == 15, "non-square (6x15) должен быть валиден")
        g.add(arr.padded_shape() == (8, 16), f"padded_shape() должен дополнить до 2^n, получено {arr.padded_shape()}")
        return g

    def test_invalid_modulation_raises(self) -> AssertionGroup:
        g = AssertionGroup("project_config.invalid_modulation_raises")
        raised = False
        try:
            ProjectConfig(modulation="fm")
        except ValueError:
            raised = True
        g.add(raised, "modulation вне {'lfm','am'} должна кидать ValueError")
        return g

    def test_invalid_am_step_raises(self) -> AssertionGroup:
        g = AssertionGroup("project_config.invalid_am_step_raises")
        raised = False
        try:
            ProjectConfig(am_step=7)
        except ValueError:
            raised = True
        g.add(raised, "am_step вне {8,16,32,64} должна кидать ValueError")
        return g

    def test_yaml_roundtrip_via_run_workspace(self) -> AssertionGroup:
        """A5: ProjectConfig грузится через run_workspace (to_yaml/from_yaml), не YamlConfigSource."""
        g = AssertionGroup("project_config.yaml_roundtrip")
        cfg = ProjectConfig(
            array=ArrayConfig(nx=6, ny=15), modulation="am", am_window_depth=32, am_step=16, n_pulses=32,
        )
        text = to_yaml(project_config_to_dict(cfg))
        restored = project_config_from_dict(from_yaml(text))
        g.add(restored.array.nx == 6 and restored.array.ny == 15, "array переживает YAML-роундтрип")
        g.add(restored.modulation == "am", "modulation переживает YAML-роундтрип")
        g.add(restored.am_window_depth == 32, "am_window_depth переживает YAML-роундтрип")
        g.add(restored.am_step == 16, "am_step переживает YAML-роундтрип")
        g.add(restored.n_pulses == 32, "n_pulses переживает YAML-роундтрип")
        g.add(restored.wave.fs == cfg.wave.fs, "wave (WaveTimeConfig) переживает YAML-роундтрип")
        return g


class MessageBusTests(TestRunner):

    def setup(self) -> None:
        self.bus = MessageBus()
        self.received: list[tuple[str, object]] = []

        class _Obs:
            def on_data(_self, key: str, data: object) -> None:  # noqa: N805
                self.received.append((key, data))

        self.observer = _Obs()

    def test_publish_notifies_subscriber(self) -> AssertionGroup:
        g = AssertionGroup("message_bus.publish_notifies_subscriber")
        self.bus.subscribe("tracks", self.observer)
        self.bus.publish("tracks", {"x": 1})
        g.add(len(self.received) == 1, f"observer должен получить 1 уведомление, получено {len(self.received)}")
        g.add(self.received[0] == ("tracks", {"x": 1}), "observer должен получить (key, data)")
        return g

    def test_unsubscribe_stops_notifications(self) -> AssertionGroup:
        g = AssertionGroup("message_bus.unsubscribe_stops_notifications")
        self.bus.subscribe("cube", self.observer)
        self.bus.unsubscribe("cube", self.observer)
        self.bus.publish("cube", 123)
        g.add(len(self.received) == 0, "после unsubscribe наблюдатель не должен получать данные")
        return g

    def test_publish_without_subscribers_does_not_raise(self) -> AssertionGroup:
        g = AssertionGroup("message_bus.publish_without_subscribers")
        raised = False
        try:
            self.bus.publish("squares", None)
        except Exception:  # noqa: BLE001
            raised = True
        g.add(not raised, "publish в канал без подписчиков не должен кидать исключение")
        return g

    def test_data_context_composes_bus_and_keeps_save_load(self) -> AssertionGroup:
        """F1: DataContext композирует MessageBus, save_cube/load_cube не сломаны."""
        g = AssertionGroup("message_bus.data_context_composition")
        dc = DataContext(root="/tmp/radar3d_test_body_motion_dc")
        dc.subscribe("tracks", self.observer)
        dc.publish("tracks", "hello")
        g.add(len(self.received) == 1, "DataContext.publish должен уведомить наблюдателя через шину")

        cube = np.arange(8, dtype=np.complex64).reshape(2, 2, 2)
        path = dc.save_cube("bm_p1_smoke", cube)
        loaded = dc.load_cube("bm_p1_smoke")
        g.add(bool(np.array_equal(cube, loaded)), "save_cube/load_cube должны остаться рабочими (F1)")
        g.add(isinstance(path, str), "save_cube должен вернуть путь")
        return g


class MotionModelTests(TestRunner):

    def setup(self) -> None:
        self.dt = 1.0
        self.n_tacts = 60
        self.initial = TargetState(pos=np.array([0.0, 0.0, -8000.0]), vel=np.array([60.0, 5.0, 180.0]))

    def _run(self, model, rng) -> list[TargetState]:
        states = [self.initial]
        state = self.initial
        for _ in range(self.n_tacts):
            state = model.propagate(state, self.dt, rng)
            states.append(state)
        return states

    def test_constant_velocity_is_exact_line(self) -> AssertionGroup:
        g = AssertionGroup("motion.constant_velocity_exact_line")
        states = self._run(ConstantVelocity(), np.random.default_rng(0))
        expected_final = self.initial.pos + self.initial.vel * self.dt * self.n_tacts
        g.add(np.allclose(states[-1].pos, expected_final, atol=1e-6),
              f"cv: pos({self.n_tacts}) должен быть pos0+vel*n*dt, получено {states[-1].pos} vs {expected_final}")
        g.add(states[-1].tact == self.n_tacts, "tact должен инкрементироваться на каждый propagate")
        return g

    def test_markov_drift_within_aero_limits_and_smooth(self) -> AssertionGroup:
        g = AssertionGroup("motion.markov_drift_limits_and_smoothness")
        model = MarkovDrift(max_turn_rate=0.03, max_accel=0.6)
        rng = np.random.default_rng(11)
        states = self._run(model, rng)

        max_heading_jump = 0.0
        max_speed_jump = 0.0
        for prev, cur in zip(states[:-1], states[1:], strict=True):
            prev_speed = float(np.linalg.norm(prev.vel))
            cur_speed = float(np.linalg.norm(cur.vel))
            max_speed_jump = max(max_speed_jump, abs(cur_speed - prev_speed))
            if prev_speed > 1e-6 and cur_speed > 1e-6:
                cos_angle = float(np.dot(prev.vel, cur.vel) / (prev_speed * cur_speed))
                angle = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                max_heading_jump = max(max_heading_jump, angle)

        g.add(max_heading_jump < 0.15,
              f"курс не должен скакать резко (без рывков), max_heading_jump={max_heading_jump:.4f} рад")
        g.add(max_speed_jump < model.max_accel * self.dt + 1e-6,
              f"скорость не должна скакать резче клипа аэро-лимита, max_speed_jump={max_speed_jump:.3f}")

        positions = np.array([s.pos for s in states])
        step_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        g.add(float(step_lengths.std()) < float(step_lengths.mean()) + 1e-3,
              "длины шагов траектории не должны иметь выбросов (гладкость)")
        return g

    def test_coordinated_turn_is_wide_radius(self) -> AssertionGroup:
        g = AssertionGroup("motion.coordinated_turn_wide_radius")
        model = CoordinatedTurn(turn_rate=0.01)
        rng = np.random.default_rng(0)
        states = self._run(model, rng)
        speeds = [float(np.linalg.norm(s.vel)) for s in states]
        g.add(max(speeds) - min(speeds) < 1e-6, "координированный вираж не должен менять модуль скорости")

        max_heading_jump = 0.0
        for prev, cur in zip(states[:-1], states[1:], strict=True):
            cos_angle = float(np.dot(prev.vel, cur.vel) /
                               (np.linalg.norm(prev.vel) * np.linalg.norm(cur.vel)))
            max_heading_jump = max(max_heading_jump, float(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
        g.add(max_heading_jump < 0.05, f"вираж должен быть широким (без рывков), получено {max_heading_jump:.4f}")
        return g

    def test_constant_accel_changes_speed_monotonically(self) -> AssertionGroup:
        g = AssertionGroup("motion.constant_accel_monotonic_speed")
        model = ConstantAccel(accel_along_track=1.0, max_accel=3.0)
        rng = np.random.default_rng(0)
        states = self._run(model, rng)
        speeds = [float(np.linalg.norm(s.vel)) for s in states]
        g.add(all(b >= a - 1e-9 for a, b in zip(speeds[:-1], speeds[1:], strict=True)),
              "разгон (accel_along_track>0) должен монотонно увеличивать скорость")
        return g


class KinematicsTests(TestRunner):

    def setup(self) -> None:
        self.cfg = ProjectConfig()
        self.kin = Kinematics(self.cfg)

    def test_vr_sign_approaching_vs_receding(self) -> AssertionGroup:
        g = AssertionGroup("kinematics.vr_sign")
        approaching = TargetState(pos=[0.0, 0.0, -5000.0], vel=[0.0, 0.0, 200.0])
        receding = TargetState(pos=[0.0, 0.0, -5000.0], vel=[0.0, 0.0, -200.0])
        sample_app = self.kin.project(approaching)
        sample_rec = self.kin.project(receding)
        g.add(sample_app.vr < 0, f"приближение должно давать vr<0, получено {sample_app.vr}")
        g.add(sample_rec.vr > 0, f"удаление должно давать vr>0, получено {sample_rec.vr}")
        return g

    def test_kx_ky_within_aperture_for_boresight_and_small_angle(self) -> AssertionGroup:
        g = AssertionGroup("kinematics.kx_ky_within_aperture")
        boresight = TargetState(pos=[0.0, 0.0, -5000.0], vel=[0.0, 0.0, 100.0])
        sample = self.kin.project(boresight)
        g.add(abs(sample.kx) < 1e-6, f"на нормали kx должен быть ~0, получено {sample.kx}")
        g.add(abs(sample.ky) < 1e-6, f"на нормали ky должен быть ~0, получено {sample.ky}")

        off_axis = TargetState(pos=[600.0, 300.0, -5000.0], vel=[0.0, 0.0, 100.0])
        sample2 = self.kin.project(off_axis)
        half_nx, half_ny = self.cfg.array.nx / 2.0, self.cfg.array.ny / 2.0
        g.add(abs(sample2.kx) <= half_nx, f"kx должен оставаться в пределах апертуры, получено {sample2.kx}")
        g.add(abs(sample2.ky) <= half_ny, f"ky должен оставаться в пределах апертуры, получено {sample2.ky}")
        return g

    def test_range_bin_matches_range_resolution(self) -> AssertionGroup:
        g = AssertionGroup("kinematics.range_bin_matches_resolution")
        state = TargetState(pos=[0.0, 0.0, -3000.0], vel=[0.0, 0.0, 0.0])
        sample = self.kin.project(state)
        expected_bin = 3000.0 / self.kin._range_resolution  # noqa: SLF001 -- прямая сверка формулы в тесте
        g.add(abs(sample.range_bin - expected_bin) < 1e-6,
              f"range_bin должен быть r/range_resolution, получено {sample.range_bin} vs {expected_bin}")
        return g

    def test_tact_sequence_publishes_track_and_advances_state(self) -> AssertionGroup:
        g = AssertionGroup("kinematics.tact_sequence_track")
        dc = DataContext(root="/tmp/radar3d_test_body_motion_tacts")
        received: list[Tact] = []

        class _Obs:
            def on_data(self, key: str, data: object) -> None:
                received.append(data)  # type: ignore[arg-type]

        dc.subscribe("tracks", _Obs())
        initial = TargetState(pos=[0.0, 0.0, -4000.0], vel=[40.0, 0.0, 120.0])
        seq = TactSequence(initial, MarkovDrift(), self.kin, n_tacts=8, dt=1.0,
                            rng=np.random.default_rng(5), data_context=dc)
        tacts = list(seq)
        g.add(len(tacts) == 8, f"итератор должен отдать n_tacts=8 записей, получено {len(tacts)}")
        g.add(len(received) == 8, f"шина должна получить 8 публикаций, получено {len(received)}")
        g.add(tacts[0].state.tact == 0, "первый такт должен иметь state.tact=0")
        g.add(tacts[-1].state.tact == 7, f"такты должны идти по порядку, последний={tacts[-1].state.tact}")
        return g


if __name__ == "__main__":
    ok = True
    for cls in [ProjectConfigTests, MessageBusTests, MotionModelTests, KinematicsTests]:
        ok = cls().run_all() and ok
    sys.exit(0 if ok else 1)
