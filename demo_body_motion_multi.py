"""Демо P4 body-motion: несколько целей (Composite-сцена) + разные законы движения + помехи P3.

Мульти-трек (M2/M4, `MemoryBank/tasks/TASK_body_motion_p4.md`): 3 цели -- каждая со СВОИМ
`MotionModel` (`ConstantVelocity`/`MarkovDrift`/`CoordinatedTurn`) и своим seed
(`TargetHandle`, `core/generators/tact_sequence.py`). `MultiTactSequence` двигает ВСЕ
состояния одновременно на каждом такте (Q8: "всё меняется каждый такт"), `iter_multi_cubes`
(P4/M1/M3, `core/generators/volume.py`) даёт когерентную сумму целей БЕЗ шума + шум ОДИН
раз поверх суммы (не N раз); помехи (P3, `SceneModeler.contribute_to`) накладываются
поверх -- тот же порядок вызовов builder->modeler, что уже в `demo_body_motion_jammers.py`
(P4 не дублирует, а комбинирует P2/P3/P4-строительные блоки).

Ориентация осей (конвенция P1/P2/P3): дальность R -- в горизонте, kx (азимут) -- вбок,
ky (угол места) -- вверх.

Выход:
    graphics/body_motion/p4_multi/multi_tracks.png   -- 3D-траектории всех целей (цветные)
    graphics/body_motion/p4_multi/multi_tracks.gif   -- анимация одновременного движения
    graphics/body_motion/p4_multi/angle_map.png      -- угловая карта последнего такта:
                                                         N пиков целей + метки помех (P3)

Запуск:
    python demo_body_motion_multi.py
    python demo_body_motion_multi.py --no-gif
    python demo_body_motion_multi.py --seed 7
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from core.config import BarrageSpec, DrfmCombSpec, JammerFlags, ProjectConfig, SceneConfig  # noqa: E402
from core.generators import (  # noqa: E402
    MultiTactSequence,
    SceneModeler,
    TargetHandle,
    VolumeBuilder,
    iter_multi_cubes,
)
from core.graphics import FigureWriter  # noqa: E402
from core.motion import (  # noqa: E402
    ConstantVelocity,
    CoordinatedTurn,
    Kinematics,
    MarkovDrift,
    MotionModel,
    TargetState,
)

OUT_DIR = os.path.join("graphics", "body_motion", "p4_multi")
N_TACTS = 24
N_SAMPLES = 1024
SNR_DB = 18.0
PULSE_FRAC = 0.05
DT = 1.0

# К3-подобно (см. scene_modeler.py): мощности помех -- параметры ДЕМО-сцены, не хардкод логики.
BARRAGE_SPEC = BarrageSpec(kx=-7.0, ky=6.0, power=25.0)
COMB_SPEC = DrfmCombSpec(kx=7.0, ky=-6.0, amplitude=6.0,
                          lead_bin=40.0, spacing=60.0, count=5, decay=0.85)

_BG = "#0d1117"
_FG = "#c9d1d9"
_TARGET_COLORS = ("#58a6ff", "#3fb950", "#f0883e")
_TARGET_LABELS = ("цель 1: ConstantVelocity", "цель 2: MarkovDrift", "цель 3: CoordinatedTurn")
_BARRAGE_COLOR = "#f85149"
_COMB_COLOR = "#d29922"
C_LIGHT = 299_792_458.0


def _target_states(rng: np.random.Generator) -> list[TargetState]:
    """3 старта, разнесённые по квадрантам апертуры и дальности -- пики не перекрываются."""
    bands = (
        dict(r0=(7000.0, 9000.0), x0=(1800.0, 2600.0), y0=(600.0, 1400.0)),
        dict(r0=(7500.0, 9500.0), x0=(-2600.0, -1800.0), y0=(-1400.0, -600.0)),
        dict(r0=(6500.0, 8500.0), x0=(-300.0, 300.0), y0=(1800.0, 2600.0)),
    )
    states = []
    for band in bands:
        r0 = float(rng.uniform(*band["r0"]))
        x0 = float(rng.uniform(*band["x0"]))
        y0 = float(rng.uniform(*band["y0"]))
        vz = float(rng.uniform(110.0, 160.0))
        vx = float(rng.uniform(-8.0, 8.0))
        vy = float(rng.uniform(-5.0, 5.0))
        states.append(TargetState(pos=np.array([x0, y0, -r0]), vel=np.array([vx, vy, vz])))
    return states


def _style_axes(ax, r_all: np.ndarray, kx_all: np.ndarray, ky_all: np.ndarray) -> None:
    """Тот же приём осей, что `demo_body_motion.py::_style_axes` (P1) -- не дублируем логику
    расчёта пределов, только копия маленькой функции стилизации (модуль P1 -- скрипт, не либа)."""
    ax.set_facecolor(_BG)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(_BG)
        pane.pane.set_alpha(1.0)
    ax.set_xlabel("дальность R, м  (горизонт ->)")
    ax.set_ylabel("kx (азимут)")
    ax.set_zlabel("ky (угол места)")
    ax.set_xlim(r_all.min(), r_all.max())
    pad_y = max(1.0, 0.15 * (kx_all.max() - kx_all.min() + 1e-6))
    pad_z = max(1.0, 0.15 * (ky_all.max() - ky_all.min() + 1e-6))
    ax.set_ylim(kx_all.min() - pad_y, kx_all.max() + pad_y)
    ax.set_zlim(ky_all.min() - pad_z, ky_all.max() + pad_z)
    ax.view_init(elev=22, azim=-72)


def _angular_energy_db(vol: np.ndarray) -> np.ndarray:
    """`mean|FFT2(апертура*Hamming)|**2` по дальности -- окно Хэмминга гасит sinc-лепестки
    (SPEC §5: "окно Хэмминга по апертуре ОБЯЗАТЕЛЬНО"), N пиков целей читаются чисто."""
    nx, ny, _n = vol.shape
    win = np.outer(np.hamming(nx), np.hamming(ny))
    ang = np.fft.fftshift(np.fft.fft2(vol * win[:, :, None], axes=(0, 1)), axes=(0, 1))
    energy = np.mean(np.abs(ang) ** 2, axis=2)
    return 10.0 * np.log10(energy + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 body-motion multi-target demo.")
    parser.add_argument("--no-gif", action="store_true", help="не создавать GIF (только PNG)")
    parser.add_argument("--seed", type=int, default=None,
                        help="фиксировать ГСЧ (по умолчанию случайно -> каждый запуск другой)")
    args = parser.parse_args()

    cfg = ProjectConfig(scene=SceneConfig(
        jammers=JammerFlags(barrage=True, comb=True),
        barrage_spec=BARRAGE_SPEC,
        comb_spec=COMB_SPEC,
    ))
    kinematics = Kinematics(cfg)

    seed_seq = np.random.SeedSequence(args.seed)
    setup_seed, build_seed, jam_seed, *target_seeds = seed_seq.spawn(3 + len(_TARGET_COLORS))
    rng_setup = np.random.default_rng(setup_seed)

    inits = _target_states(rng_setup)
    models: list[MotionModel] = [ConstantVelocity(), MarkovDrift(max_turn_rate=0.03, max_accel=0.6),
                                 CoordinatedTurn(turn_rate=0.015)]
    targets = [TargetHandle(init, model, seed=int(seed.generate_state(1)[0]))
               for init, model, seed in zip(inits, models, target_seeds, strict=True)]

    for init, label in zip(inits, _TARGET_LABELS, strict=True):
        print(f"{label}: старт pos={np.round(init.pos, 0)} vel={np.round(init.vel, 1)}")

    seq = MultiTactSequence(targets, kinematics, n_tacts=N_TACTS, dt=DT)
    builder = VolumeBuilder(n_samples=N_SAMPLES, snr_db=SNR_DB, pulse_frac=PULSE_FRAC, dt=DT)
    modeler = SceneModeler()

    build_rng = np.random.default_rng(build_seed)
    jam_rng = np.random.default_rng(jam_seed)
    multi_tacts = []
    cubes = []
    for multi_tact, vol in iter_multi_cubes(seq, builder, cfg, build_rng):
        vol_j = modeler.contribute_to(vol, cfg, jam_rng)
        multi_tacts.append(multi_tact)
        cubes.append(vol_j)
    print(f"Тактов: {len(multi_tacts)}, целей: {len(targets)}")

    # --- треки по (r, kx, ky) на цель -----------------------------------------
    n_targets = len(targets)
    tracks_r = [np.array([mt.tacts[i].sample.r for mt in multi_tacts]) for i in range(n_targets)]
    tracks_kx = [np.array([mt.tacts[i].sample.kx for mt in multi_tacts]) for i in range(n_targets)]
    tracks_ky = [np.array([mt.tacts[i].sample.ky for mt in multi_tacts]) for i in range(n_targets)]

    r_all = np.concatenate(tracks_r)
    kx_all = np.concatenate(tracks_kx)
    ky_all = np.concatenate(tracks_ky)

    plt.rcParams.update({
        "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
        "text.color": _FG, "axes.edgecolor": _FG, "axes.labelcolor": _FG,
        "xtick.color": _FG, "ytick.color": _FG,
    })
    writer = FigureWriter(OUT_DIR)

    # --- PNG: 3D-траектории всех целей -----------------------------------------
    fig = plt.figure(figsize=(9.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax, r_all, kx_all, ky_all)
    for i in range(n_targets):
        color = _TARGET_COLORS[i % len(_TARGET_COLORS)]
        ax.plot(tracks_r[i], tracks_kx[i], tracks_ky[i], color=color, linewidth=2.2,
                label=_TARGET_LABELS[i])
        ax.scatter([tracks_r[i][0]], [tracks_kx[i][0]], [tracks_ky[i][0]], color=color,
                   s=55, marker="o", edgecolors="white", linewidths=0.6, depthshade=False)
        ax.scatter([tracks_r[i][-1]], [tracks_kx[i][-1]], [tracks_ky[i][-1]], color=color,
                   s=75, marker="^", edgecolors="white", linewidths=0.6, depthshade=False)
    ax.set_title("P4: несколько независимых целей (мульти-трек, дальность в горизонте)",
                color=_FG)
    legend = ax.legend(facecolor=_BG, edgecolor=_FG, fontsize=8, loc="upper left")
    for text in legend.get_texts():
        text.set_color(_FG)
    fig.tight_layout()
    png_path = writer.write(fig, "multi_tracks.png")
    plt.close(fig)
    print(f"PNG записан: {png_path}")

    # --- GIF: анимация одновременного движения всех целей -----------------------
    if not args.no_gif:
        anim_fig = plt.figure(figsize=(9.5, 7.0))
        aax = anim_fig.add_subplot(111, projection="3d")
        _style_axes(aax, r_all, kx_all, ky_all)
        aax.set_title("P4: цели в движении (одновременно, дальность в горизонте)", color=_FG)
        trails = []
        heads = []
        for i in range(n_targets):
            color = _TARGET_COLORS[i % len(_TARGET_COLORS)]
            (trail,) = aax.plot([], [], [], color=color, linewidth=2.2, label=_TARGET_LABELS[i])
            (head,) = aax.plot([], [], [], color=color, marker="o", markersize=8,
                               linestyle="None", markeredgecolor="white")
            trails.append(trail)
            heads.append(head)
        legend = aax.legend(facecolor=_BG, edgecolor=_FG, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(_FG)

        def _init():
            for trail, head in zip(trails, heads, strict=True):
                trail.set_data_3d([], [], [])
                head.set_data_3d([], [], [])
            return (*trails, *heads)

        def _update(frame: int):
            for i, (trail, head) in enumerate(zip(trails, heads, strict=True)):
                trail.set_data_3d(tracks_r[i][:frame + 1], tracks_kx[i][:frame + 1],
                                  tracks_ky[i][:frame + 1])
                head.set_data_3d([tracks_r[i][frame]], [tracks_kx[i][frame]], [tracks_ky[i][frame]])
            return (*trails, *heads)

        anim = FuncAnimation(anim_fig, _update, frames=N_TACTS, init_func=_init,
                             interval=140, blit=False)
        gif_path = os.path.join(OUT_DIR, "multi_tracks.gif")
        anim.save(gif_path, writer=PillowWriter(fps=8))
        plt.close(anim_fig)
        print(f"GIF записан: {gif_path}")

    # --- PNG: угловая карта последнего такта (N пиков целей + помехи) -----------
    hero_vol = cubes[-1]
    hero_tact = multi_tacts[-1]
    energy_db = _angular_energy_db(hero_vol)
    nx, ny = cfg.array.nx, cfg.array.ny
    kx_vals = np.arange(nx) - nx // 2
    ky_vals = np.arange(ny) - ny // 2
    extent = (kx_vals.min() - 0.5, kx_vals.max() + 0.5, ky_vals.min() - 0.5, ky_vals.max() + 0.5)

    fig_m = plt.figure(figsize=(6.5, 5.5))
    ax_m = fig_m.add_subplot(111)
    im = ax_m.imshow(energy_db.T, origin="lower", extent=extent, aspect="auto", cmap="turbo")
    for i, tact in enumerate(hero_tact.tacts):
        color = _TARGET_COLORS[i % len(_TARGET_COLORS)]
        ax_m.scatter([tact.sample.kx], [tact.sample.ky], s=90, facecolors="none",
                     edgecolors=color, linewidths=2.0)
        ax_m.annotate(f"ц{i + 1}", (tact.sample.kx, tact.sample.ky), color=color, fontsize=9,
                     xytext=(4, 4), textcoords="offset points")
    for label, (kx, ky, color) in {
        "заград": (BARRAGE_SPEC.kx, BARRAGE_SPEC.ky, _BARRAGE_COLOR),
        "гребёнка": (COMB_SPEC.kx, COMB_SPEC.ky, _COMB_COLOR),
    }.items():
        ax_m.scatter([kx], [ky], s=90, facecolors="none", edgecolors=color, linewidths=2.0)
        ax_m.annotate(label, (kx, ky), color=color, fontsize=9, xytext=(4, 4),
                     textcoords="offset points")
    ax_m.set_facecolor(_BG)
    ax_m.set_xlabel("kx (азимут)")
    ax_m.set_ylabel("ky (угол места)")
    ax_m.set_title(f"P4: угловая карта, такт {hero_tact.tacts[0].state.tact}, "
                   f"{n_targets} цели + заград/гребёнка", fontsize=9, color=_FG)
    cb = fig_m.colorbar(im, ax=ax_m, shrink=0.85)
    cb.set_label("мощность, дБ (mean по дальности)", color=_FG)
    cb.ax.yaxis.set_tick_params(color=_FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=_FG)
    ax_m.tick_params(colors=_FG)
    fig_m.tight_layout()
    png_angle = writer.write(fig_m, "angle_map.png")
    plt.close(fig_m)
    print(f"PNG записан: {png_angle}")


if __name__ == "__main__":
    main()
