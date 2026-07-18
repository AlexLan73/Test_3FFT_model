"""Приёмка ex2 через `common.runner.TestRunner` (🚫 pytest, правило 04).

Лёгкие размеры (R5 спеки): 16×16×512, 2 объекта (am 8п + radio 16п), depths 16/8 --
полный 64×64×4096 гоняет только сам demo-прогон, не тесты.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работала форма `python demo/tests/test_ex2.py` (конвенция репо).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import ArrayConfig, ProjectConfig  # noqa: E402
from core.generators.backends import NumpyBackend  # noqa: E402
from core.generators.waveforms import AmToCube, WaveformFactory  # noqa: E402
from demo.ex2_am_square.example import (  # noqa: E402
    Ex2AmSquare,
    Ex2Params,
    ObjectSpec,
    _object_spec,
    add_noise_volume,
    build_clean_volume,
    coarse_burst_points,
    detect_objects,
    match_metrics,
    merge_rois,
)

_TEST_SCENE = (
    ObjectSpec("X", "am", 8, 100, 3, -2),
    ObjectSpec("Y", "radio", 16, 300, -4, 5),
)

_TEST_PARAMS = Ex2Params(
    # ⚠️ Девиация от буквы ТЗ («глубины 16/8»): `AmToCube.__post_init__` (core, не трогаем)
    # жёстко требует `16 <= depth <= 256` -- depth=8 невозможен ни при каких обстоятельствах.
    # Взяты те же пропорции, что в основном прогоне (coarse=32/fine=16), но nx/ny/n_axis
    # уменьшены (R5: лёгкие размеры для тестов).
    nx=16, ny=16, n_axis=512, fs=500e6, f_m=100e6, m=0.5, env_frac=1.0 / 8.0,
    snr_db_list=(float("inf"), 10.0),
    coarse_depth=32, coarse_step=32, fine_depth=16, fine_step=8,
    pfa=1e-3, seed=7, scene=_TEST_SCENE, angle_tol=2,
)


class Ex2AmSquareTests(TestRunner):

    def setup(self) -> None:
        self.p = _TEST_PARAMS
        self.cfg = ProjectConfig(array=ArrayConfig(self.p.nx, self.p.ny), modulation="am")

    def test_volume_shape(self) -> AssertionGroup:
        g = AssertionGroup("ex2.volume_shape")
        rng = np.random.default_rng(self.p.seed)
        clean = build_clean_volume(self.p, rng)
        g.add(clean.shape == (self.p.nx, self.p.ny, self.p.n_axis),
              f"shape должен быть {(self.p.nx, self.p.ny, self.p.n_axis)}, получено {clean.shape}")
        g.add(clean.dtype == np.complex64, f"dtype должен быть complex64, получено {clean.dtype}")

        before = clean.copy()
        noisy = add_noise_volume(clean, 10.0, np.random.default_rng(1))
        g.add(np.array_equal(clean, before), "add_noise_volume не должен мутировать вход")
        g.add(noisy.shape == clean.shape, "форма шумного объёма должна совпадать с чистым")
        g.add(not np.array_equal(noisy, clean), "шумный объём должен отличаться от чистого")
        return g

    def test_steering_peak(self) -> AssertionGroup:
        g = AssertionGroup("ex2.steering_peak")
        obj = self.p.scene[0]
        rng = np.random.default_rng(self.p.seed)
        spec, modulation = _object_spec(self.p, obj)
        field_ = WaveformFactory().create(modulation).render(NumpyBackend(), spec, rng)

        start = max(0, obj.t0 - self.p.fine_depth // 2)
        cube = AmToCube(depth=self.p.fine_depth, step=self.p.fine_step, start=start).fill(
            field_.data, self.cfg,
        )
        power = cube.magnitude.astype(np.float64) ** 2
        idx = np.unravel_index(int(np.argmax(power)), power.shape)
        kx = float(cube.kx.values[idx[0]])
        ky = float(cube.ky.values[idx[1]])
        g.add(abs(kx - obj.kx) <= 1, f"пик kx={kx} должен быть у объекта kx={obj.kx} (±1 бин)")
        g.add(abs(ky - obj.ky) <= 1, f"пик ky={ky} должен быть у объекта ky={obj.ky} (±1 бин)")
        return g

    def test_coarse_finds_objects(self) -> AssertionGroup:
        g = AssertionGroup("ex2.coarse_finds_objects")
        clean = build_clean_volume(self.p, np.random.default_rng(self.p.seed))
        points = coarse_burst_points(clean, self.cfg, self.p)
        rois = merge_rois(points, self.p.coarse_step, self.p.coarse_depth)
        g.add(len(rois) >= 1, f"должен быть хотя бы 1 ROI, получено {len(rois)}")
        for obj in self.p.scene:
            covered = any(r0 <= obj.t0 < r1 for r0, r1 in rois)
            g.add(covered, f"ROI должны накрывать объект {obj.name} (t0={obj.t0}), rois={rois}")
        return g

    def test_full_pipeline_snr10(self) -> AssertionGroup:
        g = AssertionGroup("ex2.full_pipeline_snr10")
        clean = build_clean_volume(self.p, np.random.default_rng(self.p.seed))
        noisy = add_noise_volume(clean, 10.0, np.random.default_rng(self.p.seed + 1))
        dets = detect_objects(noisy, self.cfg, self.p)
        m = match_metrics(dets, self.p)
        g.add(m["found"] == 2, f"found должен быть 2, получено {m['found']} (metrics={m})")
        g.add(m["false"] == 0, f"false должен быть 0, получено {m['false']} (metrics={m})")
        return g

    def test_metrics_present(self) -> AssertionGroup:
        g = AssertionGroup("ex2.metrics_present")
        ex = Ex2AmSquare(params=self.p)
        rep = ex.run(save=False)
        g.add(bool(rep.metrics), "metrics не должны быть пустыми")
        first = next(iter(rep.metrics.values()), {})
        g.add("found" in first, f"metrics должны содержать 'found', получено {first}")
        g.add("false" in first, f"metrics должны содержать 'false', получено {first}")
        g.add("contrast_db_mean" in first, f"metrics должны содержать 'contrast_db_mean', получено {first}")
        return g


if __name__ == "__main__":
    Ex2AmSquareTests().run_all()
