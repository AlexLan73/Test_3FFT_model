"""Демо P1 body-motion: ProjectConfig -> TactSequence(MotionModel) -> 3D-траектория.

Ориентация осей: **дальность R -- в горизонте** (уходит вдаль по полу сцены),
kx (азимут) -- вбок по горизонтали, ky (угол места) -- вверх.

Выход:
    graphics/body_motion/p1_trajectory/trajectory.png   -- статичный кадр
    graphics/body_motion/p1_trajectory/trajectory.gif   -- анимация движения по тактам
    graphics/body_motion/p1_trajectory/trajectory.html  -- интерактив (если есть plotly)

Запуск:
    python demo_body_motion.py            # PNG + GIF (headless)
    python demo_body_motion.py --live     # живое окно с анимацией (нужен GUI-бэкенд)
    python demo_body_motion.py --no-gif   # только PNG (быстро)
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

_LIVE = "--live" in sys.argv
if not _LIVE:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from core.config import ProjectConfig  # noqa: E402
from core.data_context import DataContext  # noqa: E402
from core.generators import TactSequence  # noqa: E402
from core.graphics import FigureWriter  # noqa: E402
from core.motion import ConstantVelocity, Kinematics, TargetState, WeavingManeuver  # noqa: E402

OUT_DIR = os.path.join("graphics", "body_motion", "p1_trajectory")
N_TACTS = 90
DT = 1.0
SEED = None   # None -> случайный старт каждый запуск (ГСЧ)

_BG = "#0d1117"
_FG = "#c9d1d9"
_TRACK_COLOR = "#58a6ff"
_REF_COLOR = "#8b949e"
_START_COLOR = "#3fb950"
_FINISH_COLOR = "#f85149"


def _random_initial_state(rng: np.random.Generator) -> TargetState:
    """Случайная точка старта + случайный вектор скорости (каждый запуск разный)."""
    r0 = float(rng.uniform(7000.0, 10000.0))     # старт далеко
    x0 = float(rng.uniform(-2200.0, 2200.0))     # боковое смещение
    y0 = float(rng.uniform(200.0, 2600.0))       # высота
    vz = float(rng.uniform(100.0, 185.0))        # скорость сближения (+Z к решётке)
    vx = float(rng.uniform(-16.0, 16.0))
    vy = float(rng.uniform(-9.0, 9.0))
    return TargetState(pos=np.array([x0, y0, -r0]), vel=np.array([vx, vy, vz]))


def _random_maneuver(rng: np.random.Generator) -> WeavingManeuver:
    """Случайные амплитуды/периоды манёвра -- каждый полёт выглядит иначе."""
    return WeavingManeuver(
        az_amp=float(rng.uniform(0.30, 0.75)), az_period=float(rng.uniform(16.0, 28.0)),
        el_amp=float(rng.uniform(0.15, 0.38)), el_period=float(rng.uniform(10.0, 20.0)),
        speed_amp=float(rng.uniform(20.0, 48.0)), speed_period=float(rng.uniform(20.0, 34.0)))


def _run_track(model, kinematics: Kinematics, init: TargetState,
               rng: np.random.Generator,
               data_context: DataContext | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq = TactSequence(init, model, kinematics, n_tacts=N_TACTS, dt=DT,
                        rng=rng, data_context=data_context)
    kx, ky, r = [], [], []
    for tact in seq:
        kx.append(tact.sample.kx)
        ky.append(tact.sample.ky)
        r.append(tact.sample.r)
    return np.array(kx), np.array(ky), np.array(r)


class _TrackObserver:
    def __init__(self) -> None:
        self.received: list[object] = []

    def on_data(self, key: str, data: object) -> None:
        if key == "tracks":
            self.received.append(data)


def _style_axes(ax, r_all: np.ndarray, kx_all: np.ndarray, ky_all: np.ndarray) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 body-motion demo (range on horizon).")
    parser.add_argument("--live", action="store_true", help="живое окно с анимацией")
    parser.add_argument("--no-gif", action="store_true", help="не создавать GIF (только PNG)")
    parser.add_argument("--seed", type=int, default=None,
                        help="фиксировать ГСЧ (по умолчанию случайно -> каждый запуск другой)")
    args = parser.parse_args()

    cfg = ProjectConfig()
    data = DataContext(root=os.path.join("out", "data"))
    observer = _TrackObserver()
    data.subscribe("tracks", observer)

    kinematics = Kinematics(cfg)

    rng = np.random.default_rng(args.seed if args.seed is not None else SEED)
    init = _random_initial_state(rng)
    maneuver = _random_maneuver(rng)
    print(f"Старт (ГСЧ): pos={np.round(init.pos, 0)} vel={np.round(init.vel, 1)}")

    kx_cv, ky_cv, r_cv = _run_track(ConstantVelocity(), kinematics, init,
                                    np.random.default_rng(0), data_context=None)
    kx_md, ky_md, r_md = _run_track(maneuver, kinematics, init, rng, data_context=data)

    print(f"ProjectConfig: array={cfg.array}, modulation={cfg.modulation}")
    print(f"Тактов в шину (канал 'tracks'): {len(observer.received)}")
    print(f"Дальность R (горизонт), м: [{r_md.min():.1f}, {r_md.max():.1f}]")

    plt.rcParams.update({
        "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
        "text.color": _FG, "axes.edgecolor": _FG, "axes.labelcolor": _FG,
        "xtick.color": _FG, "ytick.color": _FG,
    })

    r_all = np.concatenate([r_cv, r_md])
    kx_all = np.concatenate([kx_cv, kx_md])
    ky_all = np.concatenate([ky_cv, ky_md])
    os.makedirs(OUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(9.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    _style_axes(ax, r_all, kx_all, ky_all)
    ax.plot(r_cv, kx_cv, ky_cv, color=_REF_COLOR, linestyle="--", linewidth=1.3,
            label="эталон: ConstantVelocity")
    ax.plot(r_md, kx_md, ky_md, color=_TRACK_COLOR, linewidth=2.4,
            label="манёвр: WeavingManeuver (змейка+горка+скорость)")
    ax.scatter([r_md[0]], [kx_md[0]], [ky_md[0]], color=_START_COLOR, s=70,
               label="старт", depthshade=False)
    ax.scatter([r_md[-1]], [kx_md[-1]], [ky_md[-1]], color=_FINISH_COLOR, s=70,
               label="финиш (у решётки)", depthshade=False)
    ax.set_title("P1: манёвр цели — дальность в горизонте (уклонение + смена скорости)", color=_FG)
    legend = ax.legend(facecolor=_BG, edgecolor=_FG)
    for text in legend.get_texts():
        text.set_color(_FG)
    fig.tight_layout()
    png_path = FigureWriter(OUT_DIR).write(fig, "trajectory.png")
    print(f"PNG записан: {png_path}")

    if _LIVE or not args.no_gif:
        anim_fig = plt.figure(figsize=(9.5, 7.0))
        aax = anim_fig.add_subplot(111, projection="3d")
        _style_axes(aax, r_all, kx_all, ky_all)
        aax.plot(r_cv, kx_cv, ky_cv, color=_REF_COLOR, linestyle="--", linewidth=1.0, alpha=0.5)
        aax.set_title("P1: цель в движении (дальность в горизонте)", color=_FG)
        (trail,) = aax.plot([], [], [], color=_TRACK_COLOR, linewidth=2.4)
        (head,) = aax.plot([], [], [], color=_FINISH_COLOR, marker="o", markersize=9,
                           linestyle="None")

        def _init():
            trail.set_data_3d([], [], [])
            head.set_data_3d([], [], [])
            return trail, head

        def _update(i: int):
            trail.set_data_3d(r_md[:i + 1], kx_md[:i + 1], ky_md[:i + 1])
            head.set_data_3d([r_md[i]], [kx_md[i]], [ky_md[i]])
            return trail, head

        anim = FuncAnimation(anim_fig, _update, frames=len(r_md), init_func=_init,
                             interval=80, blit=False)
        if _LIVE:
            print("Живое окно: закрой его, чтобы завершить.")
            plt.show()
        else:
            gif_path = os.path.join(OUT_DIR, "trajectory.gif")
            anim.save(gif_path, writer=PillowWriter(fps=14))
            print(f"GIF записан: {gif_path}")

    try:
        import plotly.graph_objects as go

        from core.graphics.interactive import HtmlWriter

        fig3d = go.Figure(data=[
            go.Scatter3d(x=r_md, y=kx_md, z=ky_md, mode="lines+markers",
                         line=dict(color=_TRACK_COLOR, width=5), marker=dict(size=2),
                         name="WeavingManeuver"),
            go.Scatter3d(x=r_cv, y=kx_cv, z=ky_cv, mode="lines",
                         line=dict(color=_REF_COLOR, width=2, dash="dash"),
                         name="ConstantVelocity (эталон)"),
        ])
        fig3d.update_layout(
            template="plotly_dark",
            scene=dict(xaxis_title="дальность R, м", yaxis_title="kx (азимут)",
                       zaxis_title="ky (угол места)"),
            title="P1: траектория цели (интерактив, дальность в горизонте)",
        )
        html_path = HtmlWriter(OUT_DIR).write(fig3d, "trajectory.html")
        print(f"HTML записан: {html_path}")
    except ImportError:
        print("plotly не установлен -- HTML пропущен (PNG+GIF записаны)")


if __name__ == "__main__":
    main()
