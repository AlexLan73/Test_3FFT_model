"""Тесты E3: концепция апертуры i×j (nx != ny допустим, паддинг до 2ⁿ независимо по осям).

Углового FFT-фронтенд УЖЕ переделан (E2, принят): `ArrayConfig.padded_shape()`,
`angular_fft`, `Fft3DModel`, `LfmToCube.fill` паддят угловые оси нулями до ближайшей
степени двойки НЕЗАВИСИМО по X и Y перед `fft2`/`fftn` (дальность не трогается).

Эти тесты подтверждают саму концепцию (не регресс-тест API, а proof of concept):
- `padded_shape()` считает 2ⁿ верно для неквадратных и квадратных апертур;
- `angular_fft` реально паддит нулями (а не просто обрезает/интерполирует иначе);
- `Fft3DModel`/`LfmToCube` строят оси kx/ky на padded-размерах, согласованных с
  формой спектра;
- `SpectralCube.index_of_angle` согласован с padded-сеткой (центр = N_pad//2).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_aperture_ixj.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import ArrayConfig, ProjectConfig, RangeConfig  # noqa: E402
from core.generators import ArrayGrid  # noqa: E402
from core.generators.waveforms import LfmToCube  # noqa: E402
from core.models import AxisWindows, Fft3DModel, RectWindow, angular_fft  # noqa: E402


class ApertureIxjTests(TestRunner):
    """Proof-of-concept: апертура i×j, zero-pad до 2ⁿ независимо по X и Y (F9)."""

    def test_padded_shape_non_square(self) -> AssertionGroup:
        g = AssertionGroup("aperture.padded_shape_non_square")
        g.add(ArrayConfig(6, 15).padded_shape() == (8, 16),
              f"6x15 должен паддиться до (8,16), получено {ArrayConfig(6, 15).padded_shape()}")
        g.add(ArrayConfig(16, 16).padded_shape() == (16, 16),
              f"16x16 уже 2^n -- паддинг должен быть no-op, получено {ArrayConfig(16, 16).padded_shape()}")
        g.add(ArrayConfig(3, 31).padded_shape() == (4, 32),
              f"3->4, 31->32 по независимым осям, получено {ArrayConfig(3, 31).padded_shape()}")
        g.add(ArrayConfig(33, 33).padded_shape() == (64, 64),
              f"33 (>32) должен паддиться до 64, получено {ArrayConfig(33, 33).padded_shape()}")
        return g

    def test_angular_fft_pads_non_square(self) -> AssertionGroup:
        g = AssertionGroup("aperture.angular_fft_pads_non_square")
        rng = np.random.default_rng(0)
        volume = (rng.standard_normal((6, 15, 4)) + 1j * rng.standard_normal((6, 15, 4))).astype(np.complex64)
        spectrum = angular_fft(volume)
        g.add(spectrum.shape[:2] == (8, 16),
              f"угловые оси должны паддиться до (8,16), получено {spectrum.shape[:2]}")
        g.add(spectrum.shape[2] == 4, f"дальностная ось не должна меняться, получено {spectrum.shape[2]}")
        return g

    def test_angular_fft_16_is_noop(self) -> AssertionGroup:
        g = AssertionGroup("aperture.angular_fft_16_is_noop")
        rng = np.random.default_rng(1)
        volume = (rng.standard_normal((16, 16, 4)) + 1j * rng.standard_normal((16, 16, 4))).astype(np.complex64)
        spectrum = angular_fft(volume)
        g.add(spectrum.shape == (16, 16, 4),
              f"16x16 уже 2^n -- форма не должна меняться, получено {spectrum.shape}")
        return g

    def test_lfm_to_cube_non_square(self) -> AssertionGroup:
        g = AssertionGroup("aperture.lfm_to_cube_non_square")
        cfg = replace(ProjectConfig(), array=ArrayConfig(6, 15))
        rng = np.random.default_rng(2)
        volume = (rng.standard_normal((6, 15, 256)) + 1j * rng.standard_normal((6, 15, 256))).astype(np.complex64)
        cube = LfmToCube().fill(volume, cfg)
        g.add(cube.magnitude.shape[:2] == (8, 16),
              f"magnitude должен быть паддед до (8,16) по угловым осям, получено {cube.magnitude.shape[:2]}")
        g.add(len(cube.kx.values) == 8, f"ось kx должна иметь 8 отсчётов, получено {len(cube.kx.values)}")
        g.add(len(cube.ky.values) == 16, f"ось ky должна иметь 16 отсчётов, получено {len(cube.ky.values)}")
        return g

    def test_fft3d_non_square(self) -> AssertionGroup:
        g = AssertionGroup("aperture.fft3d_non_square")
        model = Fft3DModel(ArrayConfig(6, 15), RangeConfig(64, 64),
                            windows=AxisWindows(RectWindow(), RectWindow(), RectWindow()))
        rng = np.random.default_rng(3)
        datacube = (rng.standard_normal((6, 15, 64)) + 1j * rng.standard_normal((6, 15, 64))).astype(np.complex64)
        cube = model.process(datacube)
        g.add(cube.magnitude.shape[:2] == (8, 16),
              f"Fft3DModel.process() должен паддить угловые оси до (8,16), получено {cube.magnitude.shape[:2]}")
        return g

    def test_zero_pad_values_are_zero(self) -> AssertionGroup:
        """Доказать, что паддинг -- именно нули: fft2(s=(8,16)) == fft2(np.pad(...нулями...))."""
        g = AssertionGroup("aperture.zero_pad_values_are_zero")
        rng = np.random.default_rng(4)
        volume = (rng.standard_normal((6, 15, 3)) + 1j * rng.standard_normal((6, 15, 3))).astype(np.complex64)

        via_angular_fft = angular_fft(volume, aperture_window=None)

        manual_padded = np.pad(volume, ((0, 8 - 6), (0, 16 - 15), (0, 0)))
        manual_spectrum = np.fft.fftshift(np.fft.fft2(manual_padded, axes=(0, 1)), axes=(0, 1))

        g.add(via_angular_fft.shape == manual_spectrum.shape,
              f"формы должны совпасть, получено {via_angular_fft.shape} vs {manual_spectrum.shape}")
        g.add(bool(np.allclose(via_angular_fft, manual_spectrum, atol=1e-4)),
              "angular_fft(s=(8,16)) должен совпадать с ручным np.pad нулями + fft2 (доказательство zero-pad)")
        return g

    def test_angular_index_on_padded_grid(self) -> AssertionGroup:
        """Для padded-сетки центр (боресайт kx=ky=0) должен быть на индексе N_pad//2."""
        g = AssertionGroup("aperture.angular_index_on_padded_grid")
        arr = ArrayConfig(6, 15)
        rng_cfg = RangeConfig(4, 4)
        model = Fft3DModel(arr, rng_cfg, windows=AxisWindows(RectWindow(), RectWindow(), RectWindow()))
        grid = ArrayGrid.from_config(arr)
        steering = grid.steering(0.0, 0.0)  # боресайт -- все элементы апертуры синфазны
        datacube = np.tile(steering[:, :, None], (1, 1, rng_cfg.n_real)).astype(np.complex64)
        cube = model.process(datacube)

        pow2x, pow2y = arr.padded_shape()
        plane = cube.magnitude[:, :, 0]
        ix, iy = np.unravel_index(np.argmax(plane), plane.shape)
        g.add((int(ix), int(iy)) == (pow2x // 2, pow2y // 2),
              f"пик на боресайте должен быть в центре padded-сетки {(pow2x // 2, pow2y // 2)}, "
              f"получено {(int(ix), int(iy))}")
        g.add(cube.index_of_angle(0.0, 0.0) == (pow2x // 2, pow2y // 2),
              f"index_of_angle(0,0) должен указывать на центр padded-сетки, "
              f"получено {cube.index_of_angle(0.0, 0.0)}")
        return g


if __name__ == "__main__":
    ok = ApertureIxjTests().run_all()
    sys.exit(0 if ok else 1)
