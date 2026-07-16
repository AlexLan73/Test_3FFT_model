"""Демо P3 body-motion: цель + заградительная помеха (barrage) + гребёнка DRFM (comb)
поверх сырого объёма такта.

Ориентация осей (согласовано с P1/P2, demo_body_motion.py/demo_body_motion_volume.py):
дальность (range/fast-time) -- по горизонтали. Раздельно по панелям:
  (а) 3D-скаттер В УГЛОВОЙ области (кx/ky) -- X=дальность, Y=kx (азимут), Z=ky (угол
      места). Угловая ось получена ТОЛЬКО пространственным FFT по апертуре (nx,ny)
      сырого объёма -- это НЕ полный 3D-FFT/квадрат-детектор P5 (по дальности
      остаётся сырой fast-time индекс), а лишь приём визуализации: без углового FFT
      источники (цель/заград/гребёнка) неотличимы по направлению в сыром объёме
      (см. `core/generators/scene_modeler.py`, докстринг: "полоса ... проявится
      ПОСЛЕ FFT (P5), не в P3" -- здесь эта полоса показана НАГЛЯДНО средствами
      демо, тракт/тесты P3 её не считают и не используют).
  (б) срез энергия-по-дальности: mean |vol|**2 по апертуре (nx,ny) -- РАВНО как в
      `tests/test_body_motion_jammers.py::test_target_survives_over_jammers`
      (сырой объём, без углового FFT) -- окно цели взято из `VolumeBuilder._delay_window`.
  (в) угловая карта 16x16: mean |FFT2(vol, axes=(0,1))|**2 по Z (дальности) --
      пятна заграда/гребёнки/цели на СВОИХ (kx,ky) (та же угловая проекция, что в (а),
      но свёрнутая по Z).

Порог для (а)/GIF -- НЕ константа: угловой FFT по 256 элементам апертуры поднимает
шумовой пол несогласованно (~10*log10(nx*ny) дБ), поэтому порог считается от
МЕДИАНЫ угловой энергии конкретного объёма (floor + запас), иначе при фикс. пороге
почти вся угловая карта (шумовой пол) проходит в скаттер (сотни тысяч точек, GIF
рисуется неприемлемо долго).

Реюз P1 (demo_body_motion.py): `_random_initial_state`/`_random_maneuver` -- тот же
случайный старт + манёвр (WeavingManeuver). Реюз P2 (demo_body_motion_volume.py):
`VolumeBuilder`/`iter_cubes`/стиль тёмной темы. Помехи -- `SceneModeler.contribute_to`
(P3, `core/generators/scene_modeler.py`) поверх уже построенного объёма (К1/К2/К3
сверки Кодо, TASK_body_motion_p3.md) -- мощности НЕ хардкодятся здесь в логике, а
только в параметрах демо-спек `BARRAGE_SPEC`/`COMB_SPEC` ниже (наглядность).

Выход:
    graphics/body_motion/p3_jammers/volume_jammers.png   -- (а) 3D угловой скаттер, последний такт
    graphics/body_motion/p3_jammers/range_profile.png    -- (б) энергия по дальности + окно цели
    graphics/body_motion/p3_jammers/angle_map.png        -- (в) угловая карта 16x16
    graphics/body_motion/p3_jammers/volume_jammers.gif   -- анимация (а) по тактам (опц.)

Запуск:
    python demo_body_motion_jammers.py
    python demo_body_motion_jammers.py --no-gif
    python demo_body_motion_jammers.py --seed 7
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

from core.config import (  # noqa: E402
    BarrageSpec,
    DrfmCombSpec,
    JammerFlags,
    ProjectConfig,
    SceneConfig,
)
from core.generators import SceneModeler, TactSequence, VolumeBuilder, iter_cubes  # noqa: E402
from core.graphics import FigureWriter  # noqa: E402
from core.motion import Kinematics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo_body_motion import _random_initial_state, _random_maneuver  # noqa: E402

OUT_DIR = os.path.join("graphics", "body_motion", "p3_jammers")
N_TACTS = 20
N_SAMPLES = 1024      # N по фаст-тайм -- параметр демо (см. demo_body_motion_volume.py)
SNR_DB = 15.0
PULSE_FRAC = 0.05
DT = 1.0

ANGLE_MARGIN_DB = 12.0    # порог (а)/(гиф) = медиана угловой энергии + запас (см. докстринг)

# К3 (сверка Кодо): мощности/параметры помех -- ИЗ СПЕК, наглядно заданы здесь ТОЛЬКО
# как параметры демо-сцены (не хардкод внутри SceneModeler/логики помех). kx/ky --
# целые бины (чистый пик в угловой карте), разнесены по разным квадрантам от цели.
BARRAGE_SPEC = BarrageSpec(kx=-6.0, ky=5.0, power=30.0)
COMB_SPEC = DrfmCombSpec(kx=6.0, ky=-5.0, amplitude=6.0,
                          lead_bin=40.0, spacing=60.0, count=5, decay=0.85)

_BG = "#0d1117"
_FG = "#c9d1d9"
_TARGET_COLOR = "#3fb950"
_BARRAGE_COLOR = "#f85149"
_COMB_COLOR = "#d29922"
C_LIGHT = 299_792_458.0


def _power_db(x: np.ndarray) -> np.ndarray:
    power = np.abs(x).astype(np.float64) ** 2
    return 10.0 * np.log10(power + 1e-12)


def _angular_cube(vol: np.ndarray) -> np.ndarray:
    """FFT ТОЛЬКО по апертуре (nx,ny), дальность (Z) остаётся сырым fast-time индексом.

    Приём визуализации демо (см. докстринг модуля) -- не часть тракта/тестов P3.
    """
    return np.fft.fftshift(np.fft.fft2(vol, axes=(0, 1)), axes=(0, 1))


def _style_pane(ax) -> None:
    ax.set_facecolor(_BG)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor(_BG)
        pane.pane.set_alpha(1.0)
    ax.tick_params(colors=_FG, labelsize=7)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.zaxis.label.set_color(_FG)


def _scatter_angular(ax, vol: np.ndarray, fs: float, thr_db: float, vmax_db: float):
    """(а) X=дальность, Y=kx (азимут), Z=ky (угол места) -- как в demo_body_motion.py (P1)."""
    ang = _angular_cube(vol)
    power_db = _power_db(ang)
    nx, ny, n = vol.shape
    kx_vals = np.arange(nx) - nx // 2
    ky_vals = np.arange(ny) - ny // 2
    ikx, iky, iz = np.meshgrid(kx_vals, ky_vals, np.arange(n), indexing="ij")
    mask = power_db > thr_db
    r_m = iz[mask].astype(np.float64) * (C_LIGHT / (2.0 * fs))
    sc = ax.scatter(r_m, ikx[mask], iky[mask], c=power_db[mask], cmap="turbo",
                     s=6, alpha=0.6, vmin=thr_db, vmax=vmax_db, edgecolors="none")
    _style_pane(ax)
    ax.set_xlabel("дальность (fast-time), м  (горизонт ->)", fontsize=8)
    ax.set_ylabel("kx (азимут)", fontsize=8)
    ax.set_zlabel("ky (угол места)", fontsize=8)
    ax.set_xlim(0.0, n * C_LIGHT / (2.0 * fs))
    ax.set_ylim(kx_vals.min() - 0.5, kx_vals.max() + 0.5)
    ax.set_zlim(ky_vals.min() - 0.5, ky_vals.max() + 0.5)
    ax.view_init(elev=18, azim=-60)
    return sc


def _range_profile(ax, vol: np.ndarray, fs: float, start: int, stop: int) -> None:
    """(б) mean |vol|**2 по апертуре -- дальность в горизонте, окно цели заштриховано.

    Сырой (до FFT) домен: заград/гребёнка/цель складываются как фазово-когерентные
    по апертуре, но НЕ разнесённые по направлению вклады (см. докстринг модуля) --
    в этом срезе они видны как общий подъём/биения фона, а не как изолированные
    пики (ожидаемо, разделение по направлению -- панели а/в и P5). Отмечен
    глобальный максимум энергии -- критерий приёмки ("цель выживает",
    `tests/test_body_motion_jammers.py::test_target_survives_over_jammers`)
    формально доказан ТЕСТОМ на дефолтных мощностях спек (`BarrageSpec()`/
    `DrfmCombSpec()`, К3 сверки). При наглядно завышенных мощностях демо (K3,
    "30..60") запас по этой сырой RAW-метрике заметно уже -- глобальный максимум
    энергии по всей дальности (экстремум ~N=1024 шумоподобных отсчётов) не всегда
    попадает точно в окно цели на каждой конкретной реализации шума. Это ЧЕСТНО
    показано (легенда/подпись), а не скрыто -- устойчивое отделение цели от
    заграда/гребёнки по направлению видно на панелях (а)/(в) (угловой FFT).
    """
    energy = np.mean(np.abs(vol) ** 2, axis=(0, 1))
    n = energy.shape[0]
    r_m = np.arange(n) * (C_LIGHT / (2.0 * fs))
    energy_db = 10.0 * np.log10(energy + 1e-12)
    ax.plot(r_m, energy_db, color=_FG, linewidth=0.9)
    ax.axvspan(r_m[start], r_m[stop], color=_TARGET_COLOR, alpha=0.25, label="окно цели")

    peak_in_window = float(energy[start:stop + 1].max())
    outside = np.delete(energy, np.arange(start, stop + 1))
    ratio = peak_in_window / float(np.median(outside))
    argmax = int(np.argmax(energy))
    ax.scatter([r_m[argmax]], [energy_db[argmax]], s=45, color=_TARGET_COLOR,
               edgecolors="white", linewidths=0.6, zorder=5,
               label=f"глоб. максимум (в окне: {start <= argmax <= stop})")

    ax.set_facecolor(_BG)
    ax.set_xlabel("дальность (fast-time), м")
    ax.set_ylabel("мощность, дБ (mean по апертуре)")
    ax.set_title(
        f"(б) энергия по дальности (сырой домен, до FFT): пик/медиана вне окна = {ratio:.1f}x\n"
        "заград+гребёнка поднимают общий фон по ВСЕЙ дальности (разделение по углу -- панели а/в)",
        fontsize=9, color=_FG)
    ax.legend(loc="upper right", facecolor=_BG, labelcolor=_FG, fontsize=8)
    ax.tick_params(colors=_FG)


def _angle_map(ax, vol: np.ndarray, markers: dict[str, tuple[float, float, str]]) -> None:
    """(в) 16x16 угловая карта: mean |FFT2(vol, axes=(0,1))|**2 по Z."""
    ang = _angular_cube(vol)
    nx, ny, _n = vol.shape
    energy = np.mean(np.abs(ang) ** 2, axis=2)          # (nx, ny)
    energy_db = 10.0 * np.log10(energy + 1e-12)
    kx_vals = np.arange(nx) - nx // 2
    ky_vals = np.arange(ny) - ny // 2
    extent = (kx_vals.min() - 0.5, kx_vals.max() + 0.5, ky_vals.min() - 0.5, ky_vals.max() + 0.5)
    im = ax.imshow(energy_db.T, origin="lower", extent=extent, aspect="auto",
                    cmap="turbo")
    for label, (kx, ky, color) in markers.items():
        ax.scatter([kx], [ky], s=90, facecolors="none", edgecolors=color, linewidths=2.0)
        ax.annotate(label, (kx, ky), color=color, fontsize=9, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_facecolor(_BG)
    ax.set_xlabel("kx (азимут)")
    ax.set_ylabel("ky (угол места)")
    ax.set_title("(в) угловая карта 16x16: mean|FFT2(апертура)|² по дальности", fontsize=9, color=_FG)
    ax.tick_params(colors=_FG)
    return im


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 body-motion jammers demo (barrage + comb).")
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
    builder = VolumeBuilder(n_samples=N_SAMPLES, snr_db=SNR_DB, pulse_frac=PULSE_FRAC, dt=DT)
    modeler = SceneModeler()

    seed_seq = np.random.SeedSequence(args.seed)
    setup_seed, motion_seed, build_seed, jam_seed = seed_seq.spawn(4)
    rng_setup = np.random.default_rng(setup_seed)
    init = _random_initial_state(rng_setup)
    maneuver = _random_maneuver(rng_setup)
    print(f"Старт (ГСЧ): pos={np.round(init.pos, 0)} vel={np.round(init.vel, 1)}")
    print(f"Помехи: barrage kx={BARRAGE_SPEC.kx} ky={BARRAGE_SPEC.ky} power={BARRAGE_SPEC.power} | "
          f"comb kx={COMB_SPEC.kx} ky={COMB_SPEC.ky} lead={COMB_SPEC.lead_bin} "
          f"spacing={COMB_SPEC.spacing} count={COMB_SPEC.count}")

    seq = TactSequence(init, maneuver, kinematics, n_tacts=N_TACTS, dt=DT,
                        rng=np.random.default_rng(motion_seed))
    build_rng = np.random.default_rng(build_seed)
    jam_rng = np.random.default_rng(jam_seed)
    cubes = []
    for tact, vol in iter_cubes(seq, builder, cfg, build_rng):
        vol_j = modeler.contribute_to(vol, cfg, jam_rng)
        cubes.append((tact, vol_j))
    print(f"Тактов -> кубов (с помехами): {len(cubes)}")

    fs = cfg.wave.fs
    ang_dbs = [_power_db(_angular_cube(v)) for _, v in cubes]
    ang_max_db = max(float(d.max()) for d in ang_dbs)
    ang_floor_db = float(np.median(ang_dbs[-1]))     # шумовой пол (после углового FFT, ~+10log10(nx*ny))
    angle_thr_db = ang_floor_db + ANGLE_MARGIN_DB
    r_all = [tact.sample.r for tact, _ in cubes]
    print(f"Дальность R, м: [{min(r_all):.1f}, {max(r_all):.1f}]; "
          f"угловой шумовой пол={ang_floor_db:.1f} дБ, порог скаттера={angle_thr_db:.1f} дБ "
          f"(max={ang_max_db:.1f})")

    plt.rcParams.update({
        "figure.facecolor": _BG, "axes.facecolor": _BG, "savefig.facecolor": _BG,
        "text.color": _FG, "axes.edgecolor": _FG, "axes.labelcolor": _FG,
        "xtick.color": _FG, "ytick.color": _FG,
    })
    writer = FigureWriter(OUT_DIR)

    hero_tact, hero_vol = cubes[-1]
    window = builder._delay_window(hero_tact.sample.r, cfg.wave.fs)  # noqa: SLF001 -- как в тестах P3
    mask = window.mask(builder.n_samples, cfg.wave.fs)
    idx = np.flatnonzero(mask)
    start, stop = int(idx[0]), int(idx[-1])

    markers = {
        "цель": (hero_tact.sample.kx, hero_tact.sample.ky, _TARGET_COLOR),
        "заград": (BARRAGE_SPEC.kx, BARRAGE_SPEC.ky, _BARRAGE_COLOR),
        "гребёнка": (COMB_SPEC.kx, COMB_SPEC.ky, _COMB_COLOR),
    }

    # --- (а) 3D угловой скаттер: последний такт --------------------------------
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = _scatter_angular(ax, hero_vol, fs, angle_thr_db, ang_max_db)
    ax.set_title(
        f"P3: цель + заград + гребёнка (угловой FFT по апертуре, демо-приём), "
        f"такт {hero_tact.state.tact}, R={hero_tact.sample.r:.0f} м\n"
        "заград -- полоса на своём угле по ВСЕЙ дальности; гребёнка -- цепочка пиков; "
        "цель -- компактный пик", color=_FG, fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
    cb.set_label("мощность, дБ (угловой FFT)", color=_FG)
    cb.ax.yaxis.set_tick_params(color=_FG)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=_FG)
    fig.tight_layout()
    png_a = writer.write(fig, "volume_jammers.png")
    plt.close(fig)
    print(f"PNG записан: {png_a}")

    # --- (б) энергия по дальности (mean по апертуре), сырой объём -------------
    fig_b, ax_b = plt.subplots(figsize=(9, 4.5))
    _range_profile(ax_b, hero_vol, fs, start, stop)
    fig_b.tight_layout()
    png_b = writer.write(fig_b, "range_profile.png")
    plt.close(fig_b)
    print(f"PNG записан: {png_b}")

    # --- (в) угловая карта 16x16 -----------------------------------------------
    fig_c, ax_c = plt.subplots(figsize=(6.5, 5.5))
    im = _angle_map(ax_c, hero_vol, markers)
    cb_c = fig_c.colorbar(im, ax=ax_c, shrink=0.85)
    cb_c.set_label("мощность, дБ (mean по дальности)", color=_FG)
    cb_c.ax.yaxis.set_tick_params(color=_FG)
    plt.setp(plt.getp(cb_c.ax.axes, "yticklabels"), color=_FG)
    fig_c.tight_layout()
    png_c = writer.write(fig_c, "angle_map.png")
    plt.close(fig_c)
    print(f"PNG записан: {png_c}")

    # --- GIF по тактам (динамика, а-панель) ------------------------------------
    if not args.no_gif:
        anim_fig = plt.figure(figsize=(8.5, 7))
        aax = anim_fig.add_subplot(111, projection="3d")

        def _update(i: int):
            aax.clear()
            tact, vol = cubes[i]
            _scatter_angular(aax, vol, fs, angle_thr_db, ang_max_db)
            aax.set_title(f"P3: такт {tact.state.tact}, R={tact.sample.r:.0f} м",
                          color=_FG, fontsize=9)
            return ()

        anim = FuncAnimation(anim_fig, _update, frames=len(cubes), interval=220, blit=False)
        gif_path = os.path.join(OUT_DIR, "volume_jammers.gif")
        anim.save(gif_path, writer=PillowWriter(fps=4))
        plt.close(anim_fig)
        print(f"GIF записан: {gif_path}")


if __name__ == "__main__":
    main()
