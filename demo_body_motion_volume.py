"""Демо P2 body-motion: TactSequence -> VolumeBuilder -> сырой объём (nx,ny,N) -> 3D-энергия.

Ориентация осей (согласовано с P1, demo_body_motion.py): дальность (range/fast-time)
-- по горизонтали (X, уходит вдаль), ny (апертура) -- Y, nx (апертура) -- Z.

⚠️ До FFT (P5) "идеальной точки" в кубе нет: raw steering -- это только фазовый
вектор (|steer|=1 на каждом элементе), пространственный пик по (kx,ky) появляется
ТОЛЬКО после углового FFT. В сыром объёме цель видна как "стена" повышенной энергии
по ВСЕЙ апертуре на своей дальностной позиции (splat по времени/дальности), на фоне
теплового шума. Это ожидаемо и соответствует SPEC (§2, прототип P6/P2) -- не баг.

Реюз P1 (demo_body_motion.py): `_random_initial_state`/`_random_maneuver` -- тот же
случайный старт + манёвр (WeavingManeuver), чтобы сцена была "живой" при каждом запуске.

Выход:
    graphics/body_motion/p2_volume/volume.png    -- один такт (последний), 3D-энергия
    graphics/body_motion/p2_volume/timeline.png  -- 6 тактов (видно движение "стены" по дальности)
    graphics/body_motion/p2_volume/volume.gif    -- анимация по тактам

Запуск:
    python demo_body_motion_volume.py
    python demo_body_motion_volume.py --no-gif
    python demo_body_motion_volume.py --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from core.config import ProjectConfig  # noqa: E402
from core.data_context import DataContext  # noqa: E402
from core.generators import TactSequence, VolumeBuilder, iter_cubes  # noqa: E402
from core.motion import Kinematics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_body_motion import _random_initial_state, _random_maneuver  # noqa: E402

OUT_DIR = os.path.join("graphics", "body_motion", "p2_volume")
N_TACTS = 30
N_SAMPLES = 1024         # N по фаст-тайм -- параметр демо (быстрее, чем боевые 1024..10000, SPEC §1); хватает на R P1 (7-10 км)
SNR_DB = 15.0
PULSE_FRAC = 0.05
THRESHOLD_DB = 7.0       # ~5x шумового пола (NOISE_POWER=1.0) -- цель плотная "стена", шум -- редкие точки
DT = 1.0

_BG = "#0d1117"
_FG = "#c9d1d9"
C_LIGHT = 299_792_458.0


def _power_db(vol: np.ndarray) -> np.ndarray:
    power = np.abs(vol).astype(np.float64) ** 2
    return 10.0 * np.log10(power + 1e-12)


def _style_pane(ax) -> None:
    ax.set_facecolor(_BG)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(_BG)
        pane.pane.set_alpha(1.0)
    ax.tick_params(colors=_FG, labelsize=7)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.zaxis.label.set_color(_FG)


def _scatter_cube(ax, vol: np.ndarray, fs: float, thr_db: float, vmax_db: float):
    """Воксели выше порога: X=дальность (fast-time, м), Y=ny, Z=nx; цвет/альфа по мощности, дБ."""
    power_db = _power_db(vol)
    nx, ny, n = vol.shape
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(n), indexing="ij")
    mask = power_db > thr_db
    r_m = iz[mask].astype(np.float64) * (C_LIGHT / (2.0 * fs))
    sc = ax.scatter(r_m, iy[mask], ix[mask], c=power_db[mask], cmap="turbo",
                     s=5, alpha=0.55, vmin=thr_db, vmax=vmax_db, edgecolors="none")
    _style_pane(ax)
    ax.set_xlabel("дальность (fast-time), м  (горизонт ->)", fontsize=8)
    ax.set_ylabel("ny (апертура)", fontsize=8)
    ax.set_zlabel("nx (апертура)", fontsize=8)
    ax.set_xlim(0.0, n * C_LIGHT / (2.0 * fs))
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_zlim(-0.5, nx - 0.5)
    ax.view_init(elev=18, azim=-60)
    return sc


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 body-motion volume demo (raw cube, pre-FFT).")
    parser.add_argument("--no-gif", action="store_true", help="не создавать GIF (только PNG)")
    parser.add_argument("--seed", type=int, default=None,
                        help="фиксировать ГСЧ (по умолчанию случайно -> каждый запуск другой)")
    args = parser.parse_args()

    cfg = ProjectConfig()
    data = DataContext(root=os.path.join("out", "data"))
    kinematics = Kinematics(cfg)
    builder = VolumeBuilder(n_samples=N_SAMPLES, snr_db=SNR_DB, pulse_frac=PULSE_FRAC, dt=DT)

    class _CubeObserver:
        """Наблюдатель шины "cube" (Observer, SPEC §4) -- считает публикации (доказательство сцепки)."""

        def __init__(self) -> None:
            self.n = 0

        def on_data(self, key: str, payload: object) -> None:
            if key == "cube":
                self.n += 1

    observer = _CubeObserver()
    data.subscribe("cube", observer)

    seed_seq = np.random.SeedSequence(args.seed)
    setup_seed, motion_seed, build_seed = seed_seq.spawn(3)
    rng_setup = np.random.default_rng(setup_seed)
    init = _random_initial_state(rng_setup)
    maneuver = _random_maneuver(rng_setup)
    print(f"Старт (ГСЧ): pos={np.round(init.pos, 0)} vel={np.round(init.vel, 1)}")
    print(f"ProjectConfig: modulation={cfg.modulation}, array={cfg.array}, "
          f"N={N_SAMPLES}, snr_db={SNR_DB}")

    seq = TactSequence(init, maneuver, kinematics, n_tacts=N_TACTS, dt=DT,
                        rng=np.random.default_rng(motion_seed))
    cubes = list(iter_cubes(seq, builder, cfg, np.random.default_rng(build_seed), data_context=data))
    print(f"Тактов -> кубов: {len(cubes)}; публикаций в шину 'cube' (Observer): {observer.n}")

    fs = cfg.wave.fs
    global_max_db = max(float(_power_db(v).max()) for _, v in cubes)
    r_all = [tact.sample.r for tact, _ in cubes]
    print(f"Дальность R, м: [{min(r_all):.1f}, {max(r_all):.1f}]; "
          f"порог визуала {THRESHOLD_DB:.1f} дБ, глобальный максимум {global_max_db:.1f} дБ")

    plt.rcParams.update({
        "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
        "text.color": _FG, "axes.edgecolor": _FG, "axes.labelcolor": _FG,
        "xtick.color": _FG, "ytick.color": _FG,
    })
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. "hero"-кадр: последний такт -------------------------------------
    hero_tact, hero_vol = cubes[-1]
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = _scatter_cube(ax, hero_vol, fs, THRESHOLD_DB, global_max_db)
    ax.set_title(
        f"P2: сырой объём (nx,ny,N)={hero_vol.shape}, такт {hero_tact.state.tact}, "
        f"R={hero_tact.sample.r:.0f} м\n"
        "до FFT (P5) пика по апертуре нет -- цель = 'стена' энергии на своей дальности",
        color=_FG, fontsize=9,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label("мощность, дБ (относительно шумового пола)", color=_FG)
    cb.ax.yaxis.set_tick_params(color=_FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=_FG)
    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, "volume.png")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"PNG записан: {png_path}")

    # --- 2. Таймлайн: 6 тактов -> видно движение "стены" по дальности -------
    idxs = np.linspace(0, len(cubes) - 1, 6).astype(int)
    fig2, axes = plt.subplots(2, 3, figsize=(15, 9), subplot_kw={"projection": "3d"})
    for ax2, i in zip(axes.ravel(), idxs, strict=True):
        tact, vol = cubes[i]
        _scatter_cube(ax2, vol, fs, THRESHOLD_DB, global_max_db)
        ax2.set_title(f"такт {tact.state.tact}, R={tact.sample.r:.0f} м", color=_FG, fontsize=8)
    fig2.suptitle("P2: движение цели по тактам (сырой объём, дальность в горизонте)",
                  color=_FG, fontsize=11)
    fig2.tight_layout()
    timeline_path = os.path.join(OUT_DIR, "timeline.png")
    fig2.savefig(timeline_path, dpi=110)
    plt.close(fig2)
    print(f"Таймлайн записан: {timeline_path}")

    # --- 3. GIF по тактам -----------------------------------------------------
    if not args.no_gif:
        anim_fig = plt.figure(figsize=(8.5, 7))
        aax = anim_fig.add_subplot(111, projection="3d")

        def _update(i: int):
            aax.clear()
            tact, vol = cubes[i]
            _scatter_cube(aax, vol, fs, THRESHOLD_DB, global_max_db)
            aax.set_title(f"P2: такт {tact.state.tact}, R={tact.sample.r:.0f} м", color=_FG, fontsize=9)
            return ()

        anim = FuncAnimation(anim_fig, _update, frames=len(cubes), interval=180, blit=False)
        gif_path = os.path.join(OUT_DIR, "volume.gif")
        anim.save(gif_path, writer=PillowWriter(fps=5))
        plt.close(anim_fig)
        print(f"GIF записан: {gif_path}")


if __name__ == "__main__":
    main()
