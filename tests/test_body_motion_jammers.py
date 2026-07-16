"""Тесты P3 body-motion: `SceneModeler` -- jammers-only `Scene` поверх объёма такта.

Сверка с `TASK_body_motion_p3.md` (§ "Сверка Кодо"): К1 (без ThermalNoise, только
флаги), К2 (`RangeConfig` под фактический N объёма), К3 (мощность помех -- из
дефолтов спек, не хардкод), критерии приёмки (rank-1 заграда, гребёнка -- цепочка
пиков, цель выживает, флаги вкл/выкл, входы не мутируются).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_body_motion_jammers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import (  # noqa: E402
    BarrageSpec,
    DrfmCombSpec,
    JammerFlags,
    ProjectConfig,
    RangeConfig,
    SceneConfig,
)
from core.generators import SceneModeler, VolumeBuilder  # noqa: E402
from core.generators.grid import ArrayGrid  # noqa: E402
from core.motion import Kinematics, TargetState  # noqa: E402

_JAMMER_FLAG_NAMES = ("barrage", "comb", "ham")
_UNIMPLEMENTED_FLAG_NAMES = ("cw", "vfd", "arc", "clutter")


class JammerSceneTests(TestRunner):

    def setup(self) -> None:
        self.modeler = SceneModeler()
        self.grid = ArrayGrid(16, 16)
        self.n = 512
        self.rng_range = RangeConfig(n_real=self.n, n_fft=self.n)

    def _cfg(self, flags: JammerFlags, **spec_kwargs) -> ProjectConfig:
        return ProjectConfig(scene=SceneConfig(jammers=flags, **spec_kwargs))

    def test_barrage_is_rank1(self) -> AssertionGroup:
        """К заграду (А2): вклад = steer(kx,ky) ⊗ noise -- ранг 1 по развёртке (nx*ny, N)."""
        g = AssertionGroup("jammers.barrage_rank1")
        cfg = self._cfg(JammerFlags(barrage=True),
                        barrage_spec=BarrageSpec(kx=6.0, ky=-5.0, power=40.0))
        scene = self.modeler.build_jammers(cfg)
        contrib = scene.contribute(self.grid, self.rng_range, np.random.default_rng(11))

        mat = contrib.reshape(self.grid.nx * self.grid.ny, self.n)
        sigma = np.linalg.svd(mat, compute_uv=False)
        g.add(sigma[0] > 0.0, "сигма1 должна быть положительна (есть сигнал)")
        g.add(sigma[1] < 1e-6 * sigma[0],
              f"заград должен быть rank-1: sigma2/sigma1={sigma[1] / sigma[0]:.2e}")
        return g

    def test_barrage_fills_whole_range(self) -> AssertionGroup:
        """Заград -- широкополосный шум по ВСЕЙ дальностной оси (не локализован в raw).

        Побиновая энергия |compl. gauss|**2 -- экспоненциально распределена (std/mean~=1
        для КАЖДОГО отдельного бина, это физика, а не признак нелокализованности), поэтому
        критерий "равномерность" нельзя проверять по std/mean сырых по-бинных значений.
        Правильный критерий (задача, п.1): усреднить энергию БЛОКАМИ по дальностной оси
        (блок гасит дисперсию экспоненты в ~sqrt(block) раз) и проверить равномерность
        именно блочных средних -- так видна "заливка всей дальности", а не шумовой всплеск
        в одном бине.
        """
        g = AssertionGroup("jammers.barrage_fills_range")
        cfg = self._cfg(JammerFlags(barrage=True),
                        barrage_spec=BarrageSpec(kx=2.0, ky=1.0, power=30.0))
        scene = self.modeler.build_jammers(cfg)
        contrib = scene.contribute(self.grid, self.rng_range, np.random.default_rng(3))
        energy = np.mean(np.abs(contrib) ** 2, axis=(0, 1))
        g.add(bool(np.all(energy > 0.0)), "заград должен иметь ненулевую энергию на КАЖДОМ бине дальности")

        n_blocks = 16
        block_len = self.n // n_blocks
        block_means = energy[:n_blocks * block_len].reshape(n_blocks, block_len).mean(axis=1)
        ratio = float(block_means.max() / block_means.min())
        spread = float(block_means.std() / block_means.mean())
        g.add(ratio < 2.0, f"энергия заграда должна быть равномерна по блокам дальности "
                            f"(max/min блочных средних={ratio:.2f})")
        g.add(spread < 0.3, f"разброс блочных средних энергии должен быть небольшим "
                             f"(std/mean={spread:.2f})")
        return g

    def test_comb_produces_equally_spaced_peaks(self) -> AssertionGroup:
        """DRFM-гребёнка: тоны на бинах lead+i*spacing -- проявляются после FFT вдоль Z (задача, п.5)."""
        g = AssertionGroup("jammers.comb_equally_spaced_peaks")
        lead, spacing, count = 60.0, 40.0, 5
        cfg = self._cfg(JammerFlags(comb=True),
                        comb_spec=DrfmCombSpec(kx=-6.0, ky=5.0, amplitude=9.0,
                                                lead_bin=lead, spacing=spacing, count=count, decay=0.85))
        scene = self.modeler.build_jammers(cfg)
        contrib = scene.contribute(self.grid, self.rng_range, np.random.default_rng(9))

        signal = contrib[0, 0, :]     # rank-1 (тот же аргумент, что и заград) -- любой элемент апертуры
        spectrum = np.abs(np.fft.fft(signal))
        expected_bins = sorted(int(round(lead + i * spacing)) % self.n for i in range(count))
        top_bins = sorted(int(b) for b in np.argsort(spectrum)[::-1][:count])

        g.add(len(expected_bins) >= 3, "сценарий теста должен закладывать >=3 зубца")
        g.add(top_bins == expected_bins,
              f"топ-{count} бинов спектра должны совпасть с {expected_bins}, получено {top_bins}")
        diffs = np.diff(expected_bins)
        g.add(bool(np.all(diffs == spacing)), f"зубцы должны быть равноотстоящими (шаг={spacing}): {diffs}")
        return g

    def test_target_survives_over_jammers(self) -> AssertionGroup:
        """Критерий приёмки: цель (пик энергии в своём окне) остаётся отличима поверх помех.

        К3: барраж/гребёнка -- на ДЕФОЛТАХ спек (`BarrageSpec()`/`DrfmCombSpec()`,
        `cfg.scene.*_spec=None`), т.к. именно на дефолте сверяется критерий "цель выживает".
        """
        g = AssertionGroup("jammers.target_survives")
        cfg = self._cfg(JammerFlags(barrage=True, comb=True))   # *_spec=None -> дефолты спек
        kin = Kinematics(cfg)
        builder = VolumeBuilder(n_samples=1024, snr_db=18.0, pulse_frac=0.05)
        state = TargetState(pos=np.array([500.0, 200.0, -6000.0]), vel=np.array([10.0, 5.0, 150.0]))
        sample = kin.project(state)
        vol = builder.build_from_sample(sample, cfg, np.random.default_rng(42))

        vol_j = self.modeler.contribute_to(vol, cfg, np.random.default_rng(43))

        energy = np.mean(np.abs(vol_j) ** 2, axis=(0, 1))
        window = builder._delay_window(sample.r, cfg.wave.fs)  # noqa: SLF001 -- прямая сверка окна
        mask = window.mask(builder.n_samples, cfg.wave.fs)
        idx = np.flatnonzero(mask)
        start, stop = int(idx[0]), int(idx[-1])

        peak_in_window = float(energy[start:stop + 1].max())
        outside = np.delete(energy, np.arange(start, stop + 1))
        g.add(peak_in_window > 5.0 * float(np.median(outside)),
              f"пик цели в окне ({peak_in_window:.2f}) должен заметно превышать медиану фона "
              f"вне окна ({float(np.median(outside)):.2f})")
        g.add(start <= int(np.argmax(energy)) <= stop,
              f"глобальный максимум энергии ({int(np.argmax(energy))}) должен оставаться "
              f"в окне цели [{start},{stop}] -- цель различима поверх помех")
        return g

    def test_flags_all_false_is_noop(self) -> AssertionGroup:
        g = AssertionGroup("jammers.flags_all_false_noop")
        cfg = self._cfg(JammerFlags())
        vol = self._random_volume(seed=1)
        out = self.modeler.contribute_to(vol, cfg, np.random.default_rng(2))
        g.add(bool(np.allclose(out, vol)), "при всех флагах False вклад должен быть нулевым (allclose)")
        return g

    def test_each_flag_contributes(self) -> AssertionGroup:
        g = AssertionGroup("jammers.each_flag_contributes")
        vol = self._random_volume(seed=3)
        for name in _JAMMER_FLAG_NAMES:
            cfg = self._cfg(JammerFlags(**{name: True}))
            out = self.modeler.contribute_to(vol, cfg, np.random.default_rng(4))
            g.add(not np.allclose(out, vol), f"флаг '{name}'=True должен менять объём")
        return g

    def test_unimplemented_flags_raise(self) -> AssertionGroup:
        g = AssertionGroup("jammers.unimplemented_flags_raise")
        vol = self._random_volume(seed=5)
        for name in _UNIMPLEMENTED_FLAG_NAMES:
            cfg = self._cfg(JammerFlags(**{name: True}))
            try:
                self.modeler.contribute_to(vol, cfg, np.random.default_rng(6))
            except NotImplementedError:
                g.add(True, f"'{name}' должен поднимать NotImplementedError")
            else:
                g.add(False, f"'{name}' должен поднимать NotImplementedError, но не поднял")
        return g

    def test_inputs_not_mutated(self) -> AssertionGroup:
        g = AssertionGroup("jammers.inputs_not_mutated")
        vol = self._random_volume(seed=7)
        vol_before = vol.copy()
        cfg = self._cfg(JammerFlags(barrage=True, comb=True, ham=True))
        _ = self.modeler.contribute_to(vol, cfg, np.random.default_rng(8))
        g.add(bool(np.array_equal(vol, vol_before)), "contribute_to не должен мутировать входной volume")
        return g

    def test_range_config_matches_actual_n(self) -> AssertionGroup:
        """К2: `RangeConfig` для `contribute_to` строится под ФАКТИЧЕСКИЙ N объёма, не дефолт cfg.range_."""
        g = AssertionGroup("jammers.range_config_matches_actual_n")
        cfg = self._cfg(JammerFlags(barrage=True))
        g.add(cfg.range_.n_real == 16, "дефолт cfg.range_.n_real должен остаться 16 (не трогаем ProjectConfig)")
        for n in (256, 1024):
            vol = self._random_volume(seed=13, n=n)
            out = self.modeler.contribute_to(vol, cfg, np.random.default_rng(1))
            g.add(out.shape == vol.shape, f"форма выхода должна совпасть с формой входа N={n}")
        return g

    @staticmethod
    def _random_volume(seed: int, n: int = 64) -> np.ndarray:
        rs = np.random.default_rng(seed)
        return (rs.standard_normal((16, 16, n)) + 1j * rs.standard_normal((16, 16, n))).astype(np.complex64)


if __name__ == "__main__":
    ok = JammerSceneTests().run_all()
    sys.exit(0 if ok else 1)
