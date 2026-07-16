"""Тесты P5: WaveformToCube (`LfmToCube`/`AmToCube`) + `SquareView`.

Критерии -- `MemoryBank/specs/range_scale_dictionary_2026-07-15.md` §4/§6 и
`MemoryBank/tasks/TASK_body_motion_p5.md`.

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_waveform_to_cube.py
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
from core.generators import VolumeBuilder  # noqa: E402
from core.generators.waveforms import AmToCube, LfmToCube, build_lfm_target_volume  # noqa: E402
from core.graphics import SquareView  # noqa: E402
from core.models.range_fft import C_LIGHT, RangeFft  # noqa: E402
from core.models.result import Axis, SpectralCube  # noqa: E402
from core.motion import Kinematics, TargetState  # noqa: E402


def _dr_eff(n: int, fs: float, fdev: float, r_m: float) -> float:
    """Δr_eff = c/(2·μ·T_overlap) -- эталон словаря §4 (T_overlap: сколько эха
    "накрывает" референс после дечирпа при клипе задержки в [0, (n-1)/fs])."""
    mu = fdev / (n / fs)
    max_t0 = (n - 1) / fs
    tau_true = min(2.0 * r_m / C_LIGHT, max_t0)
    start = round(tau_true * fs)
    t_overlap = (n - start) / fs
    return C_LIGHT / (2.0 * mu * t_overlap)


def _range_peak(cube: SpectralCube) -> tuple[int, float]:
    """`(iz, R_est)` пика профиля max(kx,ky) по дальности -- контрольная сверка (не SquareView)."""
    profile = cube.magnitude.max(axis=(0, 1))
    iz = int(np.argmax(profile))
    return iz, float(cube.range.values[iz])


def _minus3db_width_bins(profile: np.ndarray, iz: int) -> int:
    pk = profile[iz]
    half = pk / np.sqrt(2.0)
    left = iz
    while left > 0 and profile[left] > half:
        left -= 1
    right = iz
    while right < len(profile) - 1 and profile[right] > half:
        right += 1
    return right - left


class LfmToCubeTests(TestRunner):

    def setup(self) -> None:
        self.cfg = ProjectConfig()  # lfm, fs=12e6, carrier=2e6, fdev=6e6, array 16x16
        self.kin = Kinematics(self.cfg)
        self.n = 1024

    def _volume(self, r_m: float, snr_db: float = 25.0, seed: int = 0):
        """Цель точно по нормали (kx=ky=0), только дальность варьируется."""
        state = TargetState(pos=np.array([0.0, 0.0, -r_m]), vel=np.array([0.0, 0.0, 0.0]))
        sample = self.kin.project(state)
        vol = build_lfm_target_volume(sample, self.cfg, n_samples=self.n, snr_db=snr_db,
                                       rng=np.random.default_rng(seed))
        return sample, vol

    def test_range_estimate_within_half_bin(self) -> AssertionGroup:
        g = AssertionGroup("lfm.range_estimate_within_half_bin")
        for r_true in (2000.0, 6000.0, 9000.0):
            sample, vol = self._volume(r_true, seed=1)
            lfm = LfmToCube()
            cube = lfm.fill(vol, self.cfg)
            iz, r_est = _range_peak(cube)
            n_fft = lfm.range_fft.n_fft_for(self.n)
            mu = self.cfg.wave.fdev_hz / (self.n / self.cfg.wave.fs)
            v1 = C_LIGHT * self.cfg.wave.fs / (2.0 * mu * n_fft)
            err = abs(r_est - sample.r)
            g.add(err <= v1 / 2.0 + 1e-6,
                  f"R={sample.r:.1f}: |R_est-R_true|={err:.2f} должно быть <= V1/2={v1/2:.2f}")
        return g

    def test_width_matches_delta_r_eff(self) -> AssertionGroup:
        """Ширина -3дБ (N_fft>=4N, rect по дальности) ~ Δr_eff ± 20% (словарь §4/§6)."""
        g = AssertionGroup("lfm.width_matches_delta_r_eff")
        r_true = 3000.0
        sample, vol = self._volume(r_true, snr_db=30.0, seed=2)
        lfm = LfmToCube(range_fft=RangeFft(pad_factor=4))
        cube = lfm.fill(vol, self.cfg)
        profile = cube.magnitude.max(axis=(0, 1))
        iz = int(np.argmax(profile))
        width_bins = _minus3db_width_bins(profile, iz)
        n_fft = lfm.range_fft.n_fft_for(self.n)
        mu = self.cfg.wave.fdev_hz / (self.n / self.cfg.wave.fs)
        v1 = C_LIGHT * self.cfg.wave.fs / (2.0 * mu * n_fft)
        width_m = width_bins * v1
        dr_eff = _dr_eff(self.n, self.cfg.wave.fs, self.cfg.wave.fdev_hz, sample.r)
        rel_err = abs(width_m - dr_eff) / dr_eff
        g.add(rel_err <= 0.20,
              f"ширина -3дБ={width_m:.1f} м должна быть в пределах ±20% от Δr_eff={dr_eff:.1f} м "
              f"(отклонение {rel_err * 100:.1f}%)")
        return g

    def test_zero_pad_invariant(self) -> AssertionGroup:
        """N_fft x4 -> V1 x4, а ширина в метрах меняется НАМНОГО меньше (словарь §4 инвариант)."""
        g = AssertionGroup("lfm.zero_pad_invariant")
        r_true = 3000.0
        sample, vol = self._volume(r_true, snr_db=30.0, seed=3)
        widths_m = {}
        v1s = {}
        for pad in (4, 16):
            lfm = LfmToCube(range_fft=RangeFft(pad_factor=pad))
            cube = lfm.fill(vol, self.cfg)
            profile = cube.magnitude.max(axis=(0, 1))
            iz = int(np.argmax(profile))
            width_bins = _minus3db_width_bins(profile, iz)
            n_fft = lfm.range_fft.n_fft_for(self.n)
            mu = self.cfg.wave.fdev_hz / (self.n / self.cfg.wave.fs)
            v1 = C_LIGHT * self.cfg.wave.fs / (2.0 * mu * n_fft)
            widths_m[pad] = width_bins * v1
            v1s[pad] = v1

        v1_ratio = v1s[4] / v1s[16]
        width_ratio = widths_m[4] / widths_m[16]
        g.add(3.9 <= v1_ratio <= 4.1, f"V1(pad4)/V1(pad16) должно быть ~4, получено {v1_ratio:.2f}")
        g.add(0.5 <= width_ratio <= 2.0,
              f"ширина в метрах НЕ должна масштабироваться как V1 (x4) -- отношение "
              f"width(pad4)/width(pad16)={width_ratio:.2f} должно остаться в [0.5,2.0]")
        return g

    def test_angle_matches_kinematics(self) -> AssertionGroup:
        g = AssertionGroup("lfm.angle_matches_kinematics")
        state = TargetState(pos=np.array([900.0, 300.0, -5000.0]), vel=np.array([0.0, 0.0, 0.0]))
        sample = self.kin.project(state)
        vol = build_lfm_target_volume(sample, self.cfg, n_samples=self.n, snr_db=25.0,
                                       rng=np.random.default_rng(4))
        cube = LfmToCube().fill(vol, self.cfg)
        iz, _ = _range_peak(cube)
        plane = cube.magnitude[:, :, iz]
        ix, iy = np.unravel_index(np.argmax(plane), plane.shape)
        expected_ix = round(sample.kx) + self.cfg.array.nx // 2
        expected_iy = round(sample.ky) + self.cfg.array.ny // 2
        g.add(abs(int(ix) - expected_ix) <= 1,
              f"ix={ix} должен быть рядом с ожидаемым бином kx={expected_ix} (kx={sample.kx:.3f})")
        g.add(abs(int(iy) - expected_iy) <= 1,
              f"iy={iy} должен быть рядом с ожидаемым бином ky={expected_iy} (ky={sample.ky:.3f})")
        return g

    def test_target_confined_to_single_neighbor_block(self) -> AssertionGroup:
        """Цель = компактный пик (1 окно ±N), НЕ размазана на 51 бин / 637 м (старый баг A9-gap1)."""
        g = AssertionGroup("lfm.target_confined_to_single_neighbor_block")
        sample, vol = self._volume(4000.0, snr_db=25.0, seed=6)
        cube = LfmToCube().fill(vol, self.cfg)
        view = SquareView(neighbor_planes=5)
        ix, iy, iz = view.argmax_range(cube)
        profile = view.range_profile(cube, ix, iy)
        width_bins = _minus3db_width_bins(profile, iz)
        block = view.neighbor_block(cube, iz)
        g.add(width_bins <= block.shape[2],
              f"ширина пика {width_bins} бинов должна укладываться в блок окрестности "
              f"±5 ({block.shape[2]} бинов) -- НЕ 51-бинный старый баг")
        # энергия вне блока много меньше энергии внутри (по этой (ix,iy) ячейке)
        inside = float(np.sum(profile[max(0, iz - 5):iz + 6] ** 2))
        total = float(np.sum(profile ** 2))
        g.add(inside / total > 0.5,
              f"доля энергии в блоке ±5 ({inside / total:.2f}) должна быть > 0.5 (компактная цель)")
        return g

    def test_inputs_not_mutated(self) -> AssertionGroup:
        g = AssertionGroup("lfm.inputs_not_mutated")
        _, vol = self._volume(3500.0, seed=7)
        before = vol.copy()
        _ = LfmToCube().fill(vol, self.cfg)
        g.add(bool(np.array_equal(before, vol)), "fill() не должен мутировать входной volume")
        return g


class AmToCubeTests(TestRunner):

    def setup(self) -> None:
        self.cfg = ProjectConfig(modulation="am")
        self.kin = Kinematics(self.cfg)

    def test_fill_returns_single_window_cube(self) -> AssertionGroup:
        g = AssertionGroup("am.fill_returns_single_window_cube")
        builder = VolumeBuilder(n_samples=1024, snr_db=20.0, pulse_frac=0.05)
        state = TargetState(pos=np.array([200.0, 100.0, -3000.0]), vel=np.array([0.0, 0.0, 90.0]))
        sample = self.kin.project(state)
        vol = builder.build_from_sample(sample, self.cfg, np.random.default_rng(0))

        am = AmToCube(depth=16, step=8)
        cube = am.fill(vol, self.cfg)
        g.add(cube.magnitude.shape == (16, 16, 16),
              f"fill() -- один под-куб 16x16x16, получено {cube.magnitude.shape}")
        return g

    def test_scan_step_controls_overlap(self) -> AssertionGroup:
        g = AssertionGroup("am.scan_step_controls_overlap")
        builder = VolumeBuilder(n_samples=1024, snr_db=20.0, pulse_frac=0.05)
        state = TargetState(pos=np.array([0.0, 0.0, -3000.0]), vel=np.array([0.0, 0.0, 90.0]))
        sample = self.kin.project(state)
        vol = builder.build_from_sample(sample, self.cfg, np.random.default_rng(1))

        scan_8 = AmToCube(depth=16, step=8).scan(vol, self.cfg)
        scan_16 = AmToCube(depth=16, step=16).scan(vol, self.cfg)
        g.add(len(scan_8) > len(scan_16),
              f"меньший шаг (8) должен дать больше окон, чем шаг 16 "
              f"({len(scan_8)} vs {len(scan_16)})")
        positions_8 = [p for p, _ in scan_8]
        g.add(positions_8 == sorted(positions_8), "позиции скана должны идти по возрастанию")
        if len(positions_8) > 1:
            g.add(positions_8[1] - positions_8[0] == 8, "шаг между окнами должен быть = step")
        return g

    def test_depth_bounds_validated(self) -> AssertionGroup:
        g = AssertionGroup("am.depth_bounds_validated")
        raised_low = raised_high = raised_step = False
        try:
            AmToCube(depth=8)
        except ValueError:
            raised_low = True
        try:
            AmToCube(depth=512)
        except ValueError:
            raised_high = True
        try:
            AmToCube(depth=16, step=7)
        except ValueError:
            raised_step = True
        g.add(raised_low, "depth < 16 должен вызывать ValueError")
        g.add(raised_high, "depth > 256 должен вызывать ValueError")
        g.add(raised_step, "step не из (8,16,32,64) должен вызывать ValueError")
        return g

    def test_compact_source_gives_sharper_peak_than_noise(self) -> AssertionGroup:
        """Компактный источник -> резкий колокол; чистый шум -> плоский профиль (концентрация)."""
        g = AssertionGroup("am.compact_source_gives_sharper_peak_than_noise")
        builder = VolumeBuilder(n_samples=256, snr_db=25.0, pulse_frac=0.15)
        state = TargetState(pos=np.array([0.0, 0.0, -3000.0]), vel=np.array([0.0, 0.0, 90.0]))
        sample = self.kin.project(state)
        vol_target = builder.build_from_sample(sample, self.cfg, np.random.default_rng(2))
        rng = np.random.default_rng(3)
        vol_noise = (rng.standard_normal(vol_target.shape) +
                     1j * rng.standard_normal(vol_target.shape)).astype(np.complex64)

        am = AmToCube(depth=16, step=8, start=0)
        # окно на позиции пика цели (а не 0) -- находим окно с максимальной энергией
        scan = am.scan(vol_target, self.cfg)
        best_pos, best_cube = max(scan, key=lambda pc: float(pc[1].magnitude.max()))
        noise_cube = AmToCube(depth=16, step=8, start=best_pos).fill(vol_noise, self.cfg)

        def concentration(cube: SpectralCube) -> float:
            mag = cube.magnitude
            return float(mag.max() / (mag.mean() + 1e-12))

        c_target = concentration(best_cube)
        c_noise = concentration(noise_cube)
        g.add(c_target > c_noise,
              f"концентрация (max/mean) компактной цели ({c_target:.2f}) должна быть выше, "
              f"чем у чистого шума ({c_noise:.2f})")
        return g


class SquareViewTests(TestRunner):

    def _synthetic_cube(self) -> SpectralCube:
        nx, ny, nz = 16, 16, 32
        mag = np.full((nx, ny, nz), 0.01, dtype=np.float64)
        mag[3, 5, 20] = 10.0  # инъекция явного пика
        kx = Axis("kx", np.arange(-nx // 2, nx // 2), centered=True)
        ky = Axis("ky", np.arange(-ny // 2, ny // 2), centered=True)
        rng = Axis("range", np.arange(nz) * 5.0, centered=False)
        return SpectralCube(mag, kx, ky, rng)

    def test_reduce_square_shape_and_mode(self) -> AssertionGroup:
        g = AssertionGroup("square.reduce_square_shape_and_mode")
        cube = self._synthetic_cube()
        sq_max = SquareView(reduce_mode="max").reduce_square(cube)
        sq_sum = SquareView(reduce_mode="sum").reduce_square(cube)
        g.add(sq_max.shape == (16, 16), f"reduce_square(max) должен быть 16x16, получено {sq_max.shape}")
        g.add(sq_sum.shape == (16, 16), f"reduce_square(sum) должен быть 16x16, получено {sq_sum.shape}")
        g.add(bool(sq_max[3, 5] == 10.0), "reduce_square(max) должен сохранить пиковое значение")
        g.add(bool(sq_sum[3, 5] > sq_max[3, 5]), "reduce_square(sum) >= reduce_square(max) в ячейке пика")
        return g

    def test_argmax_range_matches_known_peak(self) -> AssertionGroup:
        g = AssertionGroup("square.argmax_range_matches_known_peak")
        cube = self._synthetic_cube()
        ix, iy, iz = SquareView().argmax_range(cube)
        g.add((ix, iy, iz) == (3, 5, 20), f"argmax_range должен найти (3,5,20), получено {(ix, iy, iz)}")
        return g

    def test_neighbor_block_clips_at_edges(self) -> AssertionGroup:
        g = AssertionGroup("square.neighbor_block_clips_at_edges")
        cube = self._synthetic_cube()
        view = SquareView(neighbor_planes=5)
        mid = view.neighbor_block(cube, 20)
        g.add(mid.shape[2] == 11, f"блок в середине должен быть 2*5+1=11, получено {mid.shape[2]}")
        edge = view.neighbor_block(cube, 0)
        g.add(edge.shape[2] == 6, f"блок у левого края должен обрезаться до 6, получено {edge.shape[2]}")
        far_edge = view.neighbor_block(cube, 31)
        g.add(far_edge.shape[2] == 6, f"блок у правого края должен обрезаться до 6, получено {far_edge.shape[2]}")
        return g

    def test_tokenize_finds_injected_peak(self) -> AssertionGroup:
        g = AssertionGroup("square.tokenize_finds_injected_peak")
        cube = self._synthetic_cube()
        tokens = SquareView().tokenize(cube, threshold_db=-10.0)
        found = [t for t in tokens if (t.ix, t.iy) == (3, 5)]
        g.add(len(found) == 1, f"должен найтись ровно 1 токен в (3,5), получено {len(found)}")
        if found:
            g.add(found[0].range_bin == 20, f"range_bin токена должен быть 20, получено {found[0].range_bin}")
        return g


if __name__ == "__main__":
    ok = True
    for cls in (LfmToCubeTests, AmToCubeTests, SquareViewTests):
        ok = cls().run_all() and ok
    sys.exit(0 if ok else 1)
