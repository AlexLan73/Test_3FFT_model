"""Тесты P2 body-motion: VolumeBuilder (заполнение куба nx x ny x N) + iter_cubes.

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_body_motion_volume.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import ProjectConfig  # noqa: E402
from core.data_context import DataContext  # noqa: E402
from core.generators import TactSequence, VolumeBuilder, iter_cubes  # noqa: E402
from core.generators.grid import ArrayGrid  # noqa: E402
from core.generators.volume import C_LIGHT  # noqa: E402
from core.motion import ConstantVelocity, Kinematics, TargetState  # noqa: E402
from core.snr import StatisticsSnrEstimator  # noqa: E402


class VolumeBuilderTests(TestRunner):

    def setup(self) -> None:
        self.cfg = ProjectConfig()   # defaults: lfm, fs=12e6, carrier=2e6, fdev=6e6, array 16x16
        self.kin = Kinematics(self.cfg)

    def test_shape_dtype(self) -> AssertionGroup:
        g = AssertionGroup("volume.shape_dtype")
        builder = VolumeBuilder(n_samples=2048, snr_db=15.0)
        state = TargetState(pos=np.array([500.0, 200.0, -4000.0]), vel=np.array([10.0, 5.0, 150.0]))
        vol = builder.build(state, self.cfg, np.random.default_rng(1))
        g.add(vol.shape == (self.cfg.array.nx, self.cfg.array.ny, 2048),
              f"форма должна быть (nx,ny,N)=(16,16,2048), получено {vol.shape}")
        g.add(vol.dtype == np.complex64, f"dtype должен быть complex64, получено {vol.dtype}")
        return g

    def test_am_branch_also_builds(self) -> AssertionGroup:
        """A4: cfg.modulation='am' -> AmWaveform (другая физика зонда), тот же формат выхода."""
        g = AssertionGroup("volume.am_branch")
        cfg_am = ProjectConfig(modulation="am")
        builder = VolumeBuilder(n_samples=1024, snr_db=15.0)
        state = TargetState(pos=np.array([200.0, 100.0, -3000.0]), vel=np.array([0.0, 0.0, 90.0]))
        vol = builder.build(state, cfg_am, np.random.default_rng(0))
        g.add(vol.shape == (16, 16, 1024), f"AM-ветка: форма (16,16,1024), получено {vol.shape}")
        g.add(vol.dtype == np.complex64, "AM-ветка: dtype complex64")
        return g

    def test_range_position_matches_kinematics(self) -> AssertionGroup:
        """Позиция splat'а по дальности (окно+пик энергии) должна следовать t0=2R/c."""
        g = AssertionGroup("volume.range_position_matches_kinematics")
        builder = VolumeBuilder(n_samples=2048, snr_db=25.0, pulse_frac=0.08)
        state = TargetState(pos=np.array([0.0, 0.0, -6000.0]), vel=np.array([0.0, 0.0, 0.0]))
        sample = self.kin.project(state)
        vol = builder.build_from_sample(sample, self.cfg, np.random.default_rng(2))

        energy = np.mean(np.abs(vol) ** 2, axis=(0, 1))
        window = builder._delay_window(sample.r, self.cfg.wave.fs)  # noqa: SLF001 -- прямая сверка окна
        mask = window.mask(builder.n_samples, self.cfg.wave.fs)
        idx = np.flatnonzero(mask)
        expected_start, expected_stop = int(idx[0]), int(idx[-1])
        expected_t0_idx = round(2.0 * sample.r / C_LIGHT * self.cfg.wave.fs)
        g.add(expected_start == expected_t0_idx,
              f"старт окна {expected_start} должен = round(2R/c*fs)={expected_t0_idx} (A9-gap1)")

        peak_idx = int(np.argmax(energy))
        g.add(expected_start - 2 <= peak_idx <= expected_stop + 2,
              f"пик энергии {peak_idx} должен попадать в окно [{expected_start},{expected_stop}] "
              f"(t0=2R/c, R={sample.r:.1f})")
        return g

    def test_steering_phase_matches_kx_ky(self) -> AssertionGroup:
        """Фазовый профиль по апертуре в бине пика должен соответствовать ArrayGrid.steering(kx,ky)."""
        g = AssertionGroup("volume.steering_phase_matches_kx_ky")
        builder = VolumeBuilder(n_samples=2048, snr_db=40.0, pulse_frac=0.08)  # высокий SNR -> шум мал
        state = TargetState(pos=np.array([900.0, 300.0, -5000.0]), vel=np.array([0.0, 0.0, 120.0]))
        sample = self.kin.project(state)
        vol = builder.build_from_sample(sample, self.cfg, np.random.default_rng(3))

        energy = np.mean(np.abs(vol) ** 2, axis=(0, 1))
        idx = int(np.argmax(energy))
        cell = vol[:, :, idx]
        expected = ArrayGrid(self.cfg.array.nx, self.cfg.array.ny).steering(sample.kx, sample.ky)

        ratio_actual = cell / cell[0, 0]
        ratio_expected = expected / expected[0, 0]
        num = np.vdot(ratio_expected.ravel(), ratio_actual.ravel())
        den = np.linalg.norm(ratio_expected) * np.linalg.norm(ratio_actual)
        cos_sim = abs(complex(num) / den)
        g.add(cos_sim > 0.98,
              f"фазовый профиль по апертуре должен совпадать со steering(kx,ky) "
              f"(cos-similarity={cos_sim:.4f})")
        return g

    def test_snr_calibration(self) -> AssertionGroup:
        g = AssertionGroup("volume.snr_calibration")
        target_snr_db = 15.0
        builder = VolumeBuilder(n_samples=4096, snr_db=target_snr_db, pulse_frac=0.1)
        state = TargetState(pos=np.array([0.0, 0.0, -4000.0]), vel=np.array([0.0, 0.0, 100.0]))
        sample = self.kin.project(state)

        window = builder._delay_window(sample.r, self.cfg.wave.fs)  # noqa: SLF001
        mask = window.mask(builder.n_samples, self.cfg.wave.fs)
        idx = np.flatnonzero(mask)
        support = slice(int(idx[0]), int(idx[-1]) + 1)

        stat = StatisticsSnrEstimator()
        measured = []
        for seed in range(8):
            vol = builder.build_from_sample(sample, self.cfg, np.random.default_rng(seed))
            sig = vol[0, 0, :]
            measured.append(stat.estimate(sig, support).snr_db)
        mean_snr = float(np.mean(measured))
        g.add(abs(mean_snr - target_snr_db) < 1.5,
              f"измеренный SNR {mean_snr:.2f} дБ должен быть близок к заданному {target_snr_db} дБ")
        return g

    def test_tracks_motion_across_tacts(self) -> AssertionGroup:
        """Центроид энергии активного окна должен монотонно двигаться вслед за R(такт) (сближение)."""
        g = AssertionGroup("volume.tracks_motion_across_tacts")
        builder = VolumeBuilder(n_samples=2048, snr_db=22.0, pulse_frac=0.06)
        state = TargetState(pos=np.array([0.0, 0.0, -9000.0]), vel=np.array([0.0, 0.0, 220.0]))
        seq = TactSequence(state, ConstantVelocity(), self.kin, n_tacts=8, dt=1.0,
                            rng=np.random.default_rng(0))

        centroids: list[float] = []
        rs: list[float] = []
        for tact, vol in iter_cubes(seq, builder, self.cfg, np.random.default_rng(1)):
            energy = np.mean(np.abs(vol) ** 2, axis=(0, 1))
            floor = float(np.median(energy))
            above = energy > floor * 3.0
            idxs = np.arange(energy.shape[0])
            centroid = float(np.sum(idxs[above] * energy[above]) / np.sum(energy[above]))
            centroids.append(centroid)
            rs.append(tact.sample.r)

        g.add(all(rs[i] > rs[i + 1] for i in range(len(rs) - 1)),
              f"R должен монотонно убывать по тактам, получено {[round(r, 1) for r in rs]}")
        g.add(all(centroids[i] > centroids[i + 1] for i in range(len(centroids) - 1)),
              f"центроид энергии по дальности должен монотонно убывать вслед за R, "
              f"получено {[round(c, 1) for c in centroids]}")
        return g

    def test_cube_published_to_bus(self) -> AssertionGroup:
        g = AssertionGroup("volume.cube_published_to_bus")
        received: list[np.ndarray] = []

        class _Obs:
            def on_data(self, key: str, data: object) -> None:
                if key == "cube":
                    received.append(data)  # type: ignore[arg-type]

        dc = DataContext(root="/tmp/radar3d_test_body_motion_volume")
        dc.subscribe("cube", _Obs())

        builder = VolumeBuilder(n_samples=512, snr_db=15.0)
        state = TargetState(pos=np.array([0.0, 0.0, -3000.0]), vel=np.array([0.0, 0.0, 80.0]))
        seq = TactSequence(state, ConstantVelocity(), self.kin, n_tacts=4, dt=1.0,
                            rng=np.random.default_rng(4))
        cubes = list(iter_cubes(seq, builder, self.cfg, np.random.default_rng(5), data_context=dc))

        g.add(len(cubes) == 4, f"iter_cubes должен отдать 4 объёма, получено {len(cubes)}")
        g.add(len(received) == 4, f"шина 'cube' должна получить 4 публикации, получено {len(received)}")
        g.add(all(c.shape == (16, 16, 512) for _, c in cubes), "все объёмы должны быть (16,16,512)")
        return g

    def test_inputs_not_mutated(self) -> AssertionGroup:
        g = AssertionGroup("volume.inputs_not_mutated")
        builder = VolumeBuilder(n_samples=512, snr_db=15.0)
        state = TargetState(pos=np.array([100.0, 50.0, -2500.0]), vel=np.array([5.0, 0.0, 70.0]))
        pos_before = state.pos.copy()
        vel_before = state.vel.copy()
        g.add(not state.pos.flags.writeable, "TargetState.pos должен оставаться read-only (VO)")

        _ = builder.build(state, self.cfg, np.random.default_rng(6))
        g.add(bool(np.array_equal(pos_before, state.pos)), "state.pos не должен мутироваться build()")
        g.add(bool(np.array_equal(vel_before, state.vel)), "state.vel не должен мутироваться build()")
        return g

    def test_delay_window_within_bounds(self) -> AssertionGroup:
        """A9-gap1: очень большая/нулевая дальность не должна выводить окно за пределы N."""
        g = AssertionGroup("volume.delay_window_within_bounds")
        builder = VolumeBuilder(n_samples=256, snr_db=15.0, pulse_frac=0.1)
        fs = self.cfg.wave.fs
        for r_m in (0.0, 1.0, 1_000_000.0):
            window = builder._delay_window(r_m, fs)  # noqa: SLF001
            mask = window.mask(builder.n_samples, fs)
            g.add(bool(mask.any()), f"окно для R={r_m} должно оставаться непустым в пределах N")
            g.add(int(np.flatnonzero(mask)[-1]) < builder.n_samples,
                  f"окно для R={r_m} не должно выходить за пределы N={builder.n_samples}")
        return g


if __name__ == "__main__":
    ok = VolumeBuilderTests().run_all()
    sys.exit(0 if ok else 1)
