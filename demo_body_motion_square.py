"""Демо P5: манёвр цели -> LfmToCube (2 FFT: дечирп+range глобально, угол поячеечно)
-> квадрат 16x16 (kx,ky) + блок окрестности ±N (SquareView). Плюс сравнение
сигнатур: цель (компактный пик) / заград (полоса) / АМ-ветка (разреженные выбросы).

Реюз P1 (demo_body_motion.py): `_random_initial_state`/`_random_maneuver` -- тот же
случайный старт + WeavingManeuver, чтобы сцена была "живой" при каждом запуске.
Реюз P2 (P5-фикс A9-gap1): вместо `VolumeBuilder`-инъекции короткого окна используется
`build_lfm_target_volume` (задержанное ЛЧМ-эхо, см. докстринг в waveform_to_cube.py) --
это и есть "P2 инъекция" из TASK, ПОЧИНЕННАЯ для дечирпа.

Ориентация: дальность (range) -- по горизонтали (профили), kx/ky -- квадрат апертуры.

Выход:
    graphics/body_motion/p5_squares/square_hero.png -- 4 панели (цель/блок ±N/заград/АМ)
    graphics/body_motion/p5_squares/timeline.png     -- стек квадратов по тактам
    graphics/body_motion/p5_squares/squares.gif      -- анимация квадрата+профиля по тактам

Запуск:
    python demo_body_motion_square.py
    python demo_body_motion_square.py --no-gif
    python demo_body_motion_square.py --seed 7
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
from core.generators import TactSequence, VolumeBuilder  # noqa: E402
from core.generators.backends import NumpyBackend  # noqa: E402
from core.generators.waveforms import (  # noqa: E402
    AmToCube,
    BarrageRfJammer,
    LfmToCube,
    TimeWindow,
    WaveformSpec,
    build_lfm_target_volume,
)
from core.graphics import SquareView  # noqa: E402
from core.motion import Kinematics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_body_motion import _random_initial_state, _random_maneuver  # noqa: E402

OUT_DIR = os.path.join("graphics", "body_motion", "p5_squares")
N_TACTS = 20
N_SAMPLES = 1024
SNR_DB = 20.0
DT = 1.0

_BG = "#0d1117"
_FG = "#c9d1d9"
_PEAK_COLOR = "#58a6ff"
_BLOCK_COLOR = "#3fb950"


def _style_ax(ax) -> None:
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_color(_FG)


def _barrage_volume(cfg: ProjectConfig, n_samples: int, snr_db: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Заградительная помеха с боресайта -- через ТОТ ЖЕ `render_pipeline`, что цель."""
    spec = WaveformSpec(
        fs=cfg.wave.fs, carrier_hz=cfg.wave.carrier_hz, n_samples=n_samples,
        fdev_hz=cfg.wave.fdev_hz, snr_db=snr_db, tau_s=0.0,
        window=TimeWindow(kind="full"),
        meta={"kx": 0.0, "ky": 0.0, "nx": float(cfg.array.nx), "ny": float(cfg.array.ny)},
    )
    field = BarrageRfJammer().render(NumpyBackend(), spec, rng)
    return field.data


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 body-motion square demo (LfmToCube).")
    parser.add_argument("--no-gif", action="store_true", help="не создавать GIF (только PNG)")
    parser.add_argument("--seed", type=int, default=None,
                        help="фиксировать ГСЧ (по умолчанию случайно -> каждый запуск другой)")
    args = parser.parse_args()

    cfg = ProjectConfig()
    kinematics = Kinematics(cfg)
    view = SquareView(neighbor_planes=cfg.viz_neighbor_planes)
    lfm = LfmToCube()

    seed_seq = np.random.SeedSequence(args.seed)
    setup_seed, motion_seed, build_seed, jam_seed, am_seed = seed_seq.spawn(5)
    rng_setup = np.random.default_rng(setup_seed)
    init = _random_initial_state(rng_setup)
    maneuver = _random_maneuver(rng_setup)
    print(f"Старт (ГСЧ): pos={np.round(init.pos, 0)} vel={np.round(init.vel, 1)}")
    print(f"ProjectConfig: modulation={cfg.modulation}, array={cfg.array}, "
          f"N={N_SAMPLES}, snr_db={SNR_DB}, neighbor_planes={cfg.viz_neighbor_planes}")

    seq = TactSequence(init, maneuver, kinematics, n_tacts=N_TACTS, dt=DT,
                        rng=np.random.default_rng(motion_seed))
    build_rng = np.random.default_rng(build_seed)

    tacts, squares, blocks, peaks, r_true_list = [], [], [], [], []
    for tact in seq:
        vol = build_lfm_target_volume(tact.sample, cfg, n_samples=N_SAMPLES, snr_db=SNR_DB,
                                       rng=build_rng)
        cube = lfm.fill(vol, cfg)
        ix, iy, iz = view.argmax_range(cube)
        square = view.reduce_square(cube)
        block = view.neighbor_block(cube, iz)
        tacts.append(tact)
        squares.append(square)
        blocks.append(block)
        peaks.append((ix, iy, iz, float(cube.range.values[iz])))
        r_true_list.append(tact.sample.r)

    r_est_list = [p[3] for p in peaks]
    errs = [abs(re - rt) for re, rt in zip(r_est_list, r_true_list, strict=True)]
    print(f"Тактов обработано: {len(tacts)}; |R_est-R_true| max={max(errs):.1f} м, "
          f"mean={np.mean(errs):.1f} м")

    # --- сигнатуры для сравнения (последний такт) ---------------------------------
    hero_cube = lfm.fill(build_lfm_target_volume(tacts[-1].sample, cfg, n_samples=N_SAMPLES,
                                                  snr_db=SNR_DB, rng=build_rng), cfg)
    hero_ix, hero_iy, hero_iz = view.argmax_range(hero_cube)
    hero_profile = view.range_profile(hero_cube, hero_ix, hero_iy)

    barrage_vol = _barrage_volume(cfg, N_SAMPLES, SNR_DB, np.random.default_rng(jam_seed))
    barrage_cube = lfm.fill(barrage_vol, cfg)
    barrage_profile = barrage_cube.magnitude.max(axis=(0, 1))

    cfg_am = ProjectConfig(modulation="am")
    am_builder = VolumeBuilder(n_samples=N_SAMPLES, snr_db=SNR_DB, pulse_frac=0.05)
    am_sample = kinematics.project(tacts[-1].state, DT)
    am_vol = am_builder.build_from_sample(am_sample, cfg_am, np.random.default_rng(am_seed))
    am_scan = AmToCube(depth=cfg_am.am_window_depth, step=cfg_am.am_step).scan(am_vol, cfg_am)
    am_positions = [pos for pos, _ in am_scan]
    am_energy = [float(c.magnitude.max()) for _, c in am_scan]

    plt.rcParams.update({
        "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
        "text.color": _FG, "axes.edgecolor": _FG, "axes.labelcolor": _FG,
        "xtick.color": _FG, "ytick.color": _FG,
    })
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. hero: 4 панели -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax_sq = axes[0, 0]
    im = ax_sq.imshow(squares[-1].T, origin="lower", cmap="turbo",
                       extent=(-cfg.array.nx // 2, cfg.array.nx // 2,
                               -cfg.array.ny // 2, cfg.array.ny // 2))
    ax_sq.scatter([hero_ix - cfg.array.nx // 2], [hero_iy - cfg.array.ny // 2],
                  facecolors="none", edgecolors=_PEAK_COLOR, s=120, linewidths=2)
    ax_sq.set_xlabel("kx (азимут)")
    ax_sq.set_ylabel("ky (угол места)")
    ax_sq.set_title("Квадрат 16x16 (reduce max по range) -- цель")
    fig.colorbar(im, ax=ax_sq, shrink=0.8, label="|·| (reduce max)")
    _style_ax(ax_sq)

    ax_rp = axes[0, 1]
    r_axis = hero_cube.range.values
    rp_db = 20.0 * np.log10(hero_profile + 1e-12)
    rp_db -= rp_db.max()
    ax_rp.plot(r_axis, rp_db, color=_PEAK_COLOR, linewidth=1.5)
    n_block = cfg.viz_neighbor_planes
    lo = max(0, hero_iz - n_block)
    hi = min(len(r_axis) - 1, hero_iz + n_block)
    ax_rp.axvspan(r_axis[lo], r_axis[hi], color=_BLOCK_COLOR, alpha=0.25,
                  label=f"блок окрестности ±{n_block}")
    ax_rp.axvline(tacts[-1].sample.r, color=_FG, linestyle=":", linewidth=1,
                  label=f"R_true={tacts[-1].sample.r:.0f} м")
    ax_rp.set_xlabel("дальность R, м  (горизонт ->)")
    ax_rp.set_ylabel("магнитуда, дБ")
    ax_rp.set_title(f"Профиль дальности (ix={hero_ix},iy={hero_iy}) -- ЦЕЛЬ = компактный пик,\n"
                     f"1 окно ±{n_block}, R_est={r_axis[hero_iz]:.0f} м")
    ax_rp.set_ylim(-40, 2)
    ax_rp.legend(fontsize=8, facecolor=_BG, edgecolor=_FG, labelcolor=_FG)
    _style_ax(ax_rp)

    ax_bar = axes[1, 0]
    bar_db = 20.0 * np.log10(barrage_profile + 1e-12)
    bar_db -= bar_db.max()
    ax_bar.plot(barrage_cube.range.values, bar_db, color="#f85149", linewidth=1.0)
    ax_bar.set_xlabel("дальность R, м  (горизонт ->)")
    ax_bar.set_ylabel("магнитуда, дБ")
    ax_bar.set_title("ЗАГРАД (BarrageRfJammer) -- широкополосный шум,\nПОЛОСА по всей дальности "
                      "(не сжимается дечирпом)")
    ax_bar.set_ylim(-40, 2)
    _style_ax(ax_bar)

    ax_am = axes[1, 1]
    ax_am.bar(am_positions, am_energy, width=cfg_am.am_step * 0.9, color="#d29922")
    ax_am.set_xlabel("старт под-куба (бин быстрого времени)")
    ax_am.set_ylabel("max |·| локального 3D-FFT (16x16xD)")
    ax_am.set_title(f"АМ-ветка (AmToCube, D={cfg_am.am_window_depth}, шаг={cfg_am.am_step}) -- "
                     "разреженные выбросы")
    _style_ax(ax_am)

    fig.suptitle("P5: LfmToCube (2 FFT) -- сигнатуры цель / заград / АМ-скан", color=_FG, fontsize=12)
    fig.tight_layout()
    hero_path = os.path.join(OUT_DIR, "square_hero.png")
    fig.savefig(hero_path, dpi=125)
    plt.close(fig)
    print(f"PNG записан: {hero_path}")

    # --- 2. timeline: стек квадратов по тактам ----------------------------------------
    idxs = np.linspace(0, len(tacts) - 1, 6).astype(int)
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
    vmax = max(float(squares[i].max()) for i in idxs)
    for ax2, i in zip(axes2.ravel(), idxs, strict=True):
        ax2.imshow(squares[i].T, origin="lower", cmap="turbo", vmin=0.0, vmax=vmax)
        ix, iy, iz, r_est = peaks[i]
        ax2.scatter([ix], [iy], facecolors="none", edgecolors=_PEAK_COLOR, s=90, linewidths=1.6)
        ax2.set_title(f"такт {tacts[i].state.tact}, R_est={r_est:.0f} м", fontsize=8, color=_FG)
        _style_ax(ax2)
    fig2.suptitle("P5: квадраты 16x16 по тактам (kx,ky) -- цель следует за манёвром",
                  color=_FG, fontsize=11)
    fig2.tight_layout()
    timeline_path = os.path.join(OUT_DIR, "timeline.png")
    fig2.savefig(timeline_path, dpi=110)
    plt.close(fig2)
    print(f"Таймлайн записан: {timeline_path}")

    # --- 3. GIF: квадрат + профиль дальности по тактам --------------------------------
    if not args.no_gif:
        anim_fig, (aax1, aax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        def _update(i: int):
            aax1.clear()
            aax2.clear()
            ix, iy, iz, r_est = peaks[i]
            aax1.imshow(squares[i].T, origin="lower", cmap="turbo")
            aax1.scatter([ix], [iy], facecolors="none", edgecolors=_PEAK_COLOR, s=90, linewidths=1.6)
            aax1.set_title(f"такт {tacts[i].state.tact}: квадрат 16x16", color=_FG, fontsize=9)
            _style_ax(aax1)

            cube_i_profile = blocks[i][ix, iy, :]
            xs = np.arange(len(cube_i_profile))
            aax2.plot(xs, cube_i_profile, color=_PEAK_COLOR, marker="o", markersize=3)
            aax2.set_title(f"блок ±{n_block} вокруг R_est={r_est:.0f} м "
                            f"(R_true={r_true_list[i]:.0f} м)", color=_FG, fontsize=9)
            aax2.set_xlabel("бин (внутри блока)")
            _style_ax(aax2)
            return ()

        anim = FuncAnimation(anim_fig, _update, frames=len(tacts), interval=200, blit=False)
        gif_path = os.path.join(OUT_DIR, "squares.gif")
        anim.save(gif_path, writer=PillowWriter(fps=5))
        plt.close(anim_fig)
        print(f"GIF записан: {gif_path}")


if __name__ == "__main__":
    main()
