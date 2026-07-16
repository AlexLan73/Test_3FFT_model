"""Тесты P4 body-motion: `MultiTactSequence` + `iter_multi_cubes` -- несколько независимых целей.

Сверка с `TASK_body_motion_p4.md` (§ "🔎 Сверка Кодо"): M1 (шум добавляется ОДИН раз,
не N раз на N целей), M2 (`MultiTactSequence` -- координатор РЯДОМ, не ломает
`TactSequence`), M3 (когерентная сумма целей -- поэлементное сложение массивов), M4
(своя `MotionModel` + свой seed на цель -- независимость треков). Критерии приёмки:
N пиков = N целей, независимость треков, публикация `cube`+`tracks` в шину, входы не
мутируются.

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_body_motion_multi.py
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
from core.generators import (  # noqa: E402
    MultiTact,
    MultiTactSequence,
    TargetHandle,
    VolumeBuilder,
    iter_multi_cubes,
)
from core.motion import ConstantVelocity, CoordinatedTurn, Kinematics, MarkovDrift, TargetState  # noqa: E402


def _nms_peaks(energy_db: np.ndarray, thr_db: float, min_dist: int = 3) -> list[tuple[int, int]]:
    """Грубый non-max-suppression поиск пиков (для теста): сорт по убыванию, жадно берём
    точки не ближе `min_dist` бинов к уже выбранным -- считает РАЗЛИЧИМЫЕ пики, а не сырые
    локальные максимумы (которые ловят соседние бины одного и того же пика)."""
    order = np.dstack(np.unravel_index(np.argsort(-energy_db.ravel()), energy_db.shape))[0]
    picked: list[tuple[int, int]] = []
    for i, j in order:
        v = energy_db[i, j]
        if v < thr_db:
            break
        if all((int(i) - pi) ** 2 + (int(j) - pj) ** 2 >= min_dist ** 2 for pi, pj in picked):
            picked.append((int(i), int(j)))
    return picked


def _angular_energy_db(vol: np.ndarray) -> np.ndarray:
    """`mean|FFT2(апертура*Hamming)|**2` по дальности -- та же идея, что demo_body_motion_jammers.py
    (в), с добавленным окном Хэмминга (SPEC §5, "окно Хэмминга по апертуре ОБЯЗАТЕЛЬНО" для
    визуально/тестово чистой точки без sinc-лепестков -- без окна сильные цели дают ложный
    "4-й пик" на соседнем бине из-за нахлёста sidelobe, что и проверено прототипированием)."""
    nx, ny, _n = vol.shape
    win = np.outer(np.hamming(nx), np.hamming(ny))
    ang = np.fft.fftshift(np.fft.fft2(vol * win[:, :, None], axes=(0, 1)), axes=(0, 1))
    energy = np.mean(np.abs(ang) ** 2, axis=2)
    return 10.0 * np.log10(energy + 1e-12)


class MultiTargetTests(TestRunner):

    def setup(self) -> None:
        self.cfg = ProjectConfig()   # defaults: lfm, array 16x16, fs=12e6
        self.kin = Kinematics(self.cfg)

    def _targets(self) -> list[TargetHandle]:
        """3 цели, разнесённые по (kx,ky,R) -- разные квадранты апертуры, разные дальности."""
        return [
            TargetHandle(TargetState(pos=np.array([3000.0, 800.0, -6000.0]),
                                      vel=np.array([0.0, 0.0, 120.0])), ConstantVelocity(), seed=1),
            TargetHandle(TargetState(pos=np.array([-3200.0, -700.0, -7000.0]),
                                      vel=np.array([0.0, 0.0, 130.0])), ConstantVelocity(), seed=2),
            TargetHandle(TargetState(pos=np.array([100.0, 3200.0, -5000.0]),
                                      vel=np.array([0.0, 0.0, 110.0])), ConstantVelocity(), seed=3),
        ]

    def test_n_peaks_equals_n_targets(self) -> AssertionGroup:
        """Критерий приёмки: N целей -> N различимых пиков в угловой энергии (устойчиво к seed)."""
        g = AssertionGroup("multi.n_peaks_equals_n_targets")
        targets = self._targets()
        builder = VolumeBuilder(n_samples=1024, snr_db=20.0, pulse_frac=0.05)
        for seed in range(5):
            seq = MultiTactSequence(targets, self.kin, n_tacts=1, dt=1.0)
            multi_tact, vol = next(iter_multi_cubes(seq, builder, self.cfg, np.random.default_rng(seed)))
            energy_db = _angular_energy_db(vol)
            thr = float(np.median(energy_db)) + 12.0
            peaks = _nms_peaks(energy_db, thr, min_dist=3)
            g.add(len(peaks) == len(targets),
                  f"seed={seed}: ожидалось {len(targets)} пика(ов), найдено {len(peaks)}: {peaks}")

            nx, ny = self.cfg.array.nx, self.cfg.array.ny
            for tact in multi_tact.tacts:
                kx_idx = int(round(tact.sample.kx)) + nx // 2
                ky_idx = int(round(tact.sample.ky)) + ny // 2
                nearest = min(peaks, key=lambda p: (p[0] - kx_idx) ** 2 + (p[1] - ky_idx) ** 2)
                dist2 = (nearest[0] - kx_idx) ** 2 + (nearest[1] - ky_idx) ** 2
                g.add(dist2 <= 1, f"seed={seed}: цель (kx_idx={kx_idx},ky_idx={ky_idx}) должна иметь "
                                   f"свой пик рядом, ближайший={nearest} (dist^2={dist2})")
        return g

    def test_targets_move_independently(self) -> AssertionGroup:
        """M2/M4: траектория цели A с фиксированным seed не зависит от присутствия/модели цели B."""
        g = AssertionGroup("multi.targets_move_independently")
        init_a = TargetState(pos=np.array([1000.0, 0.0, -5000.0]), vel=np.array([10.0, 0.0, 120.0]))
        init_b = TargetState(pos=np.array([-1000.0, 500.0, -6000.0]), vel=np.array([-5.0, 3.0, 100.0]))

        seq_alone = MultiTactSequence([TargetHandle(init_a, MarkovDrift(), seed=42)],
                                       self.kin, n_tacts=10, dt=1.0)
        pos_alone = [mt.tacts[0].state.pos.copy() for mt in seq_alone]

        seq_together = MultiTactSequence(
            [TargetHandle(init_a, MarkovDrift(), seed=42), TargetHandle(init_b, CoordinatedTurn(), seed=99)],
            self.kin, n_tacts=10, dt=1.0)
        pos_together_a = [mt.tacts[0].state.pos.copy() for mt in seq_together]

        g.add(len(pos_alone) == len(pos_together_a) == 10, "оба прогона должны дать 10 тактов")
        g.add(all(bool(np.allclose(a, b)) for a, b in zip(pos_alone, pos_together_a, strict=True)),
              "траектория цели A (свой seed) не должна зависеть от присутствия цели B")

        # разные seed/модели у B не портят A: другой сценарий с ДРУГИМ seed для B тоже не влияет.
        seq_together2 = MultiTactSequence(
            [TargetHandle(init_a, MarkovDrift(), seed=42), TargetHandle(init_b, MarkovDrift(), seed=7)],
            self.kin, n_tacts=10, dt=1.0)
        pos_together_a2 = [mt.tacts[0].state.pos.copy() for mt in seq_together2]
        g.add(all(bool(np.allclose(a, b)) for a, b in zip(pos_alone, pos_together_a2, strict=True)),
              "траектория цели A не должна зависеть от seed/модели цели B")
        return g

    def test_publishes_cube_and_tracks(self) -> AssertionGroup:
        """Критерий приёмки: `MultiTactSequence`/`iter_multi_cubes` публикуют cube+tracks в шину."""
        g = AssertionGroup("multi.publishes_cube_and_tracks")
        received_cube: list[np.ndarray] = []
        received_tracks: list[MultiTact] = []

        class _ObsCube:
            def on_data(self, key: str, data: object) -> None:
                if key == "cube":
                    received_cube.append(data)  # type: ignore[arg-type]

        class _ObsTracks:
            def on_data(self, key: str, data: object) -> None:
                if key == "tracks":
                    received_tracks.append(data)  # type: ignore[arg-type]

        dc = DataContext(root="/tmp/radar3d_test_body_motion_multi")
        dc.subscribe("cube", _ObsCube())
        dc.subscribe("tracks", _ObsTracks())

        targets = self._targets()
        n_tacts = 5
        seq = MultiTactSequence(targets, self.kin, n_tacts=n_tacts, dt=1.0, data_context=dc)
        builder = VolumeBuilder(n_samples=256, snr_db=15.0)
        cubes = list(iter_multi_cubes(seq, builder, self.cfg, np.random.default_rng(3), data_context=dc))

        g.add(len(cubes) == n_tacts, f"iter_multi_cubes должен отдать {n_tacts} объёмов, "
                                      f"получено {len(cubes)}")
        g.add(len(received_cube) == n_tacts, f"шина 'cube' должна получить {n_tacts} публикаций, "
                                              f"получено {len(received_cube)}")
        g.add(len(received_tracks) == n_tacts, f"шина 'tracks' должна получить {n_tacts} публикаций, "
                                                f"получено {len(received_tracks)}")
        g.add(all(isinstance(t, MultiTact) for t in received_tracks),
              "публикации 'tracks' должны быть MultiTact")
        g.add(all(len(t.tacts) == len(targets) for t in received_tracks),
              f"каждый MultiTact должен содержать {len(targets)} Tact (по числу целей)")
        g.add(all(c.shape == (16, 16, 256) for _, c in cubes), "все объёмы должны быть (16,16,256)")
        return g

    def test_noise_added_once_not_n_times(self) -> AssertionGroup:
        """M1: шумовой пол (участок, свободный от окон ЛЮБОЙ цели) не должен расти с числом целей."""
        g = AssertionGroup("multi.noise_added_once_not_n_times")
        builder = VolumeBuilder(n_samples=4096, snr_db=20.0, pulse_frac=0.05)
        rs = [8000.0, 8500.0, 9000.0, 9500.0]   # R достаточно велик -> t0 далеко от начала оси

        def _noise_var(n_targets: int, seed: int) -> float:
            targets = [
                TargetHandle(TargetState(pos=np.array([200.0 * i, 100.0 * i, -rs[i]]),
                                          vel=np.array([0.0, 0.0, 100.0])), ConstantVelocity(), seed=100 + i)
                for i in range(n_targets)
            ]
            seq = MultiTactSequence(targets, self.kin, n_tacts=1, dt=1.0)
            _, vol = next(iter_multi_cubes(seq, builder, self.cfg, np.random.default_rng(seed)))
            noise_only = vol[:, :, :150]   # заведомо ДО окна ЛЮБОЙ цели (t0 у всех >> 150/fs)
            return float(np.var(noise_only))

        var_1 = float(np.mean([_noise_var(1, seed) for seed in range(5)]))
        var_4 = float(np.mean([_noise_var(4, seed) for seed in range(5)]))

        g.add(abs(var_1 - 1.0) < 0.15, f"var(N=1) должна быть ~NOISE_POWER=1.0, получено {var_1:.3f}")
        g.add(abs(var_4 - 1.0) < 0.15, f"var(N=4) должна быть ~NOISE_POWER=1.0 (НЕ 4x), получено {var_4:.3f}")
        g.add(var_4 / var_1 < 1.5, f"мощность шума не должна расти с N целей (M1): "
                                    f"var(N=4)/var(N=1)={var_4 / var_1:.2f}")
        return g

    def test_inputs_not_mutated(self) -> AssertionGroup:
        g = AssertionGroup("multi.inputs_not_mutated")
        state_a = TargetState(pos=np.array([1000.0, 0.0, -5000.0]), vel=np.array([10.0, 0.0, 120.0]))
        state_b = TargetState(pos=np.array([-1200.0, 400.0, -6000.0]), vel=np.array([-5.0, 2.0, 110.0]))
        pos_a_before, vel_a_before = state_a.pos.copy(), state_a.vel.copy()
        pos_b_before, vel_b_before = state_b.pos.copy(), state_b.vel.copy()

        targets = [TargetHandle(state_a, ConstantVelocity(), seed=1),
                   TargetHandle(state_b, MarkovDrift(), seed=2)]
        seq = MultiTactSequence(targets, self.kin, n_tacts=5, dt=1.0)
        builder = VolumeBuilder(n_samples=256, snr_db=15.0)
        list(iter_multi_cubes(seq, builder, self.cfg, np.random.default_rng(1)))

        g.add(bool(np.array_equal(pos_a_before, state_a.pos)), "state_a.pos не должен мутироваться")
        g.add(bool(np.array_equal(vel_a_before, state_a.vel)), "state_a.vel не должен мутироваться")
        g.add(bool(np.array_equal(pos_b_before, state_b.pos)), "state_b.pos не должен мутироваться")
        g.add(bool(np.array_equal(vel_b_before, state_b.vel)), "state_b.vel не должен мутироваться")
        return g

    def test_empty_targets_raises(self) -> AssertionGroup:
        g = AssertionGroup("multi.empty_targets_raises")
        raised = False
        try:
            MultiTactSequence([], self.kin, n_tacts=5, dt=1.0)
        except ValueError:
            raised = True
        g.add(raised, "пустой список targets должен кидать ValueError")
        return g


if __name__ == "__main__":
    ok = MultiTargetTests().run_all()
    sys.exit(0 if ok else 1)
