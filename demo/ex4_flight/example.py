"""ex4 — летящая цель + летящий носитель гребёнки + дрейфующий barrage + АНИМАЦИЯ.

Канон — `MemoryBank/specs/demo_ex4_flight_2026-07-18.md` (решения Alex §1, R1-R6) +
`MemoryBank/tasks/TASK_demo_ex4_p1.md` (карта реюза). ВСЁ собрано из готового:
движение — `core/motion` (5 моделей, случайный выбор/старт как `demo_body_motion.py:57+`),
эхо S1/гребёнка/barrage/null/скан — ex3, трекинг — `core/models/tracking`, тёмный стиль/GIF/
plotly — приёмы `demo_body_motion.py` (_BG/#0d1117, FuncAnimation+PillowWriter, plotly_dark).

Компоновка кадра (решение Alex §1.6, тёмная): слева 3D (kx × позиция × ky) · справа
поле nx×ny со следами K=8 и номерами (кружок=цель/трек, квадрат=носитель гребёнки,
рамка=barrage) · внизу ряд срезов по трекам + параметры (6 признаков §4.11 + «летит»).

Выходы (`demo/graphics/ex4_flight/`): flight_trail.gif / flight_clean.gif (из ОДНОЙ
истории, R5) · flight_3d.html (plotly, вращать/зумить) · trajectory.png · last_frame.png.

Запуск:  .venv/Scripts/python.exe demo/ex4_flight/example.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Конвенция репо: работает форма `python demo/ex4_flight/example.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from core.config import ArrayConfig  # noqa: E402
from core.generators.waveforms import Modulation, build_pulse_echo_volume  # noqa: E402
from core.graphics import AngularMapVisualizer  # noqa: E402  (реюз не нужен напрямую — оставляем срезы energy)
from core.models.tokenizer import VolumeTokenizer  # noqa: E402
from core.models.tokenizer.arbiter import TargetDecision  # noqa: E402
from core.models.tracking import NearestNeighborTracker, Track  # noqa: E402
from core.motion import (  # noqa: E402
    ConstantAccel,
    ConstantVelocity,
    CoordinatedTurn,
    Kinematics,
    MarkovDrift,
    TargetState,
    WeavingManeuver,
)
from demo.core import DemoContext, DemoRunner  # noqa: E402
from demo.ex2_am_square.example import (  # noqa: E402
    Ex2Params,
    add_noise_volume,
    coarse_burst_points,
)
from demo.ex3_am_barrage.example import (  # noqa: E402  — конвейер ex3 (реюз)
    Ex3AmBarrage,
    Ex3Params,
    band_angle,
    build_drfm_comb_volume,
    build_jammer_volume,
    null_angle,
)
from core.generators.waveforms.waveform_to_cube import AmToCube  # noqa: E402

# ── тёмный стиль (приём demo_body_motion.py:49+, реюз палитры) ────────────────
_BG = "#0d1117"
_FG = "#c9d1d9"
_TRACK_COLOR = "#58a6ff"
_COMB_COLOR = "#f0883e"
_BARRAGE_COLOR = "#f85149"


def _apply_dark(fig: Figure) -> None:
    fig.patch.set_facecolor(_BG)
    for ax in fig.get_axes():
        ax.set_facecolor(_BG)
        for spine in getattr(ax, "spines", {}).values():
            spine.set_color(_FG)
        ax.tick_params(colors=_FG, labelsize=7)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        if hasattr(ax, "zaxis"):
            ax.zaxis.label.set_color(_FG)
        ax.title.set_color(_FG)


# ── параметры ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Ex4Params:
    """Все параметры ex4 (VO). Апертура — ПАРАМЕТР (готовность к 512×256, R3)."""

    nx: int = 64
    ny: int = 64
    n_axis: int = 4096
    tacts: int = 30
    dt: float = 1.0
    snr_db: float = 10.0
    k_trail: int = 8                      # хвост (решение Alex 7.1)
    seed: int = 7
    t0_lo: int = 400                      # маппинг r->позиция оси (дальность условная)
    t0_hi: int = 3600
    r_ref: float = 11000.0                # r_ref -> t0_hi
    barrage_start: tuple[float, float] = (5.0, 18.0)
    barrage_drift: float = 0.3            # дрейф угла барьера, бины/такт (R6, <=0.5)
    slice_depth: int = 16                 # окно среза ряда ③
    max_slices: int = 4                   # срезов в ряду (длиннейшие треки)


_MODEL_BUILDERS = (
    lambda rng: ConstantVelocity(),
    # диапазоны — образец `demo_body_motion.py:_random_markov/:_random_maneuver` (реюз паттерна)
    lambda rng: MarkovDrift(max_turn_rate=float(rng.uniform(0.045, 0.065)),
                            max_accel=float(rng.uniform(0.8, 1.6)),
                            heading_noise_std=float(rng.uniform(0.045, 0.065)),
                            speed_noise_std=float(rng.uniform(0.3, 0.6))),
    lambda rng: CoordinatedTurn(turn_rate=float(rng.uniform(0.02, 0.05)) *
                                (1.0 if rng.random() < 0.5 else -1.0)),
    lambda rng: ConstantAccel(accel_along_track=float(rng.uniform(0.5, 1.5))),
    lambda rng: WeavingManeuver(
        az_amp=float(rng.uniform(0.30, 0.75)), az_period=float(rng.uniform(16.0, 28.0)),
        el_amp=float(rng.uniform(0.15, 0.38)), el_period=float(rng.uniform(10.0, 20.0)),
        speed_amp=float(rng.uniform(20.0, 48.0)), speed_period=float(rng.uniform(20.0, 34.0))),
)


def _random_initial_state(rng: np.random.Generator) -> TargetState:
    """Случайный старт+скорость — паттерн `demo_body_motion.py:57` (реюз, атрибуция там)."""
    r0 = float(rng.uniform(7000.0, 10000.0))
    x0 = float(rng.uniform(-2200.0, 2200.0))
    y0 = float(rng.uniform(200.0, 2600.0))
    vz = float(rng.uniform(100.0, 185.0))
    vx = float(rng.uniform(-16.0, 16.0))
    vy = float(rng.uniform(-9.0, 9.0))
    return TargetState(pos=np.array([x0, y0, -r0]), vel=np.array([vx, vy, vz]))


class FlyingEntity:
    """Движущаяся сущность (цель | носитель гребёнки): случайная модель + Kinematics."""

    def __init__(self, name: str, rng: np.random.Generator, kin: Kinematics,
                 p: Ex4Params) -> None:
        self.name = name
        self._model = _MODEL_BUILDERS[int(rng.integers(len(_MODEL_BUILDERS)))](rng)
        self._state = _random_initial_state(rng)
        self._kin = kin
        self._p = p
        self.model_name = type(self._model).__name__

    def step(self, rng: np.random.Generator) -> tuple[float, float, int]:
        """Такт: propagate → project → (kx, ky, t0) c клиппингом в поле/ось."""
        p = self._p
        self._state = self._model.propagate(self._state, p.dt, rng)
        s = self._kin.project(self._state, p.dt)
        kx = float(np.clip(s.kx, -(p.nx // 2 - 2), p.nx // 2 - 2))
        ky = float(np.clip(s.ky, -(p.ny // 2 - 2), p.ny // 2 - 2))
        t0 = int(np.clip(p.t0_lo + s.r / p.r_ref * (p.t0_hi - p.t0_lo), p.t0_lo, p.t0_hi))
        return kx, ky, t0


@dataclass(frozen=True)
class TactRecord:
    """История одного такта (для кадров GIF/HTML — второй прогон запрещён, R5)."""

    tact: int
    truth: dict[str, tuple[float, float, int]]      # name -> (kx, ky, t0)
    points: list                                     # CoarsePoint после null (сцена)
    banded: bool
    band_angle: tuple[float, float] | None
    tracks: tuple[dict[str, Any], ...]               # снимки треков (№, kx, ky, история, летит)
    slices: tuple[dict[str, Any], ...]               # срезы ряда ③: №, карта, параметры


class Ex4Flight(DemoRunner):
    """Полёт цели и помех по тактам + трекинг + анимация (спека §1, ТЗ Alex)."""

    name = "ex4_flight"

    def __init__(self, params: Ex4Params | None = None) -> None:
        self._p = params or Ex4Params()
        self.seed = self._p.seed
        self._stats: dict[str, Any] = {}
        self._history: list[TactRecord] = []

    # ── сборка такта ─────────────────────────────────────────────────────────
    def _ex3(self) -> Ex3AmBarrage:
        p = self._p
        base = Ex2Params(nx=p.nx, ny=p.ny, n_axis=p.n_axis, scene=())
        return Ex3AmBarrage(params=Ex3Params(base=base))

    def _tact_volume(self, ex3: Ex3AmBarrage, tgt: tuple[float, float, int],
                     comb: tuple[float, float, int], barrage: tuple[float, float],
                     rng: np.random.Generator) -> np.ndarray:
        p = self._p
        p3 = ex3._p
        b = p3.base
        vol = build_pulse_echo_volume(                       # цель — правильное эхо S1
            Modulation.AM, fs=b.fs, carrier_hz=b.f_m, n_samples=b.n_axis,
            dur_samples=int(round(8 * b.fs / b.f_m)), t0_samples=tgt[2],
            kx=tgt[0], ky=tgt[1], nx=b.nx, ny=b.ny, rng=rng,
            extra_meta={"m": b.m, "f_m": b.f_m * b.env_frac},
        )
        p3_comb = Ex3Params(base=b, drfm_lead0=comb[2])      # гребёнка ЗА носителем
        vol = vol + build_drfm_comb_volume(p3_comb, comb[0], comb[1], rng)
        vol = vol + build_jammer_volume(p3, "barrage", Modulation.BARRAGE,
                                        barrage[0], barrage[1], rng)
        return add_noise_volume(vol.astype(np.complex64), p.snr_db, rng)

    def _slices_for_tracks(self, volume: np.ndarray, cfg, tracks: list[Track],
                           with_features: bool) -> tuple:
        """Ряд ③: 16-окно вокруг каждого трека → энергия nx×ny (+признаки §4.11).

        `with_features=False` в рядовых тактах: `VolumeTokenizer` — python-CFAR по
        65к ячеек (~10 c на срез) — гнать его 30×4 раз = минуты (диагностика: 17 мин
        прогона). Полные признаки считаются ТОЛЬКО на последнем такте (last_frame),
        в кадрах GIF срезы живые (energy), таблица признаков — финальная.
        """
        p = self._p
        out = []
        ranked = sorted((t for t in tracks if t.age >= 2),
                        key=lambda t: t.age, reverse=True)[: p.max_slices]
        for tr in ranked:
            # lead_r трека — в грубых окнах (шаг 32, см. decisions ниже) -> отсчёты оси
            pos = int(np.clip(tr.lead_r * 32, 0, p.n_axis - p.slice_depth))
            sub = AmToCube(depth=p.slice_depth, step=8, start=pos).fill(volume, cfg)
            feats: dict[str, float] = {}
            label = "-"
            if with_features:
                tokens = VolumeTokenizer(window_l=p.slice_depth).tokenize(sub)
                best = None
                for tok in tokens:
                    for pk in tok.peaks:
                        d2 = (pk.kx - tr.kx) ** 2 + (pk.ky - tr.ky) ** 2
                        if best is None or d2 < best[0]:
                            best = (d2, tok)
                if best is not None:
                    fv = best[1].f
                    feats = {name: float(getattr(fv, name)) for name in fv.__dataclass_fields__}
                    label = best[1].label
            out.append({
                "track_id": tr.track_id, "kx": tr.kx, "ky": tr.ky, "pos": pos,
                "energy_db": sub.angular_energy_db(), "features": feats,
                "label": label,
                "is_moving": tr.is_moving,
            })
        return tuple(out)

    # ── основной прогон ──────────────────────────────────────────────────────
    def run_history(self, rng: np.random.Generator) -> list[TactRecord]:
        """Прогнать конвейер по тактам и заполнить `self._history` + `self._stats`.

        Вынесено из `visualize` (спека web_panel_flight §0): история — ЕДИНСТВЕННЫЙ
        источник для всех рендереров (GIF/PNG/web, R5 — второй раз не считать);
        web-страница зовёт только этот метод, без matplotlib-рендеров.
        """
        p = self._p
        ex3 = self._ex3()
        cfg = ex3._cfg()
        kin = Kinematics(cfg)
        target = FlyingEntity("target", rng, kin, p)
        carrier = FlyingEntity("comb_carrier", rng, kin, p)
        b_kx, b_ky = p.barrage_start
        tracker = NearestNeighborTracker(gate=50.0, w_r=0.1, max_missed=3,
                                         moving_threshold=0.3)

        self._history = []
        found_tacts = 0
        band_tacts = 0
        for tact in range(p.tacts):
            tgt = target.step(rng)
            comb = carrier.step(rng)
            b_kx = float(np.clip(b_kx + rng.uniform(-p.barrage_drift, p.barrage_drift),
                                 -(p.nx // 2 - 2), p.nx // 2 - 2))
            b_ky = float(np.clip(b_ky + rng.uniform(-p.barrage_drift, p.barrage_drift),
                                 -(p.ny // 2 - 2), p.ny // 2 - 2))
            volume = self._tact_volume(ex3, tgt, comb, (b_kx, b_ky), rng)
            # ── лёгкий такт-конвейер (диагностика: полный fine-этап с VolumeTokenizer
            # стоил ~40 с/такт → 17 мин прогона). Сопровождение ведём по ДЕШЁВОЙ карте
            # грубого уровня (патент §4-бис.2а: «решает по карте токенов»); полный
            # токенизатор с признаками — ТОЛЬКО на последнем такте (таблица ряда ③).
            n_windows = p.n_axis // 32
            pts_b = coarse_burst_points(volume, cfg, ex3._p.base, max_points=4)
            angle = band_angle(pts_b, n_windows, ex3._p.band_gate_frac)
            banded = angle is not None
            nulled = null_angle(volume, *angle) if banded else volume
            pts_a = coarse_burst_points(nulled, cfg, ex3._p.base, max_points=4) if banded else pts_b
            metrics = {"band_angle": angle}
            band_tacts += int(banded)
            # чистка точек такта (ОБЩАЯ для сцены и трекера — ревью last_frame):
            # 1) остатки барьера: ±3 бина от полосы (боковики после rank-1 null);
            # 2) динамический гейт −15 дБ от пика такта (слабые боковики max_points-NMS).
            pts_use = pts_a
            if banded:
                bx, by = angle
                pts_use = [q for q in pts_a if abs(q.kx - bx) > 3 or abs(q.ky - by) > 3]
            if pts_use:
                d_max = max(q.db for q in pts_use)
                pts_use = [q for q in pts_use if q.db >= d_max - 15.0]
            best: dict[tuple[int, int, int], Any] = {}
            for q in pts_use:
                key = (int(round(q.kx / 2)), int(round(q.ky / 2)), q.pos // 32)
                if key not in best or q.db > best[key].db:
                    best[key] = q
            decisions = [TargetDecision(kx=q.kx, ky=q.ky, lead_r=q.pos // 32,
                                        decision="target", reason="single", n_false=0)
                         for q in best.values()]
            dets = list(best.values())          # для метрики hit ниже (CoarsePoint duck: kx/ky)
            tracks = tracker.update(decisions, tact)
            # цель найдена в такте?
            hit = any(abs(d.kx - tgt[0]) <= 2 and abs(d.ky - tgt[1]) <= 2 for d in dets)
            found_tacts += int(hit)
            work = volume  # срезы по подавленному не нужны: берём сырой (энергия объекта видна)
            self._history.append(TactRecord(
                tact=tact,
                truth={"target": tgt, "comb": comb, "barrage": (b_kx, b_ky, 0)},
                points=pts_use,                      # чищеные точки (сцена = трекер)
                banded=banded, band_angle=metrics.get("band_angle"),
                tracks=tuple({"id": t.track_id, "kx": t.kx, "ky": t.ky,
                              "lead_r": t.lead_r, "is_moving": t.is_moving,
                              "history": tuple(t.history)} for t in tracks),
                slices=self._slices_for_tracks(work, cfg, list(tracks),
                                               with_features=(tact == p.tacts - 1)),
            ))

        self._stats = {
            "tacts": p.tacts,
            "target_found": f"{found_tacts}/{p.tacts}",
            "band_detected_tacts": f"{band_tacts}/{p.tacts}",
            "models": f"target={target.model_name}, comb={carrier.model_name}",
            "tracks_final": len(self._history[-1].tracks) if self._history else 0,
        }
        return self._history

    def visualize(self, ctx: DemoContext) -> dict[str, Figure]:
        p = self._p
        self.run_history(ctx.rng)
        figures: dict[str, Figure] = {}
        figures["last_frame"] = self._compose_frame(len(self._history) - 1, trail=p.k_trail)
        figures["trajectory"] = self._fig_trajectory()
        self._write_gifs_and_html()
        return figures

    def report_metrics(self, ctx: DemoContext) -> dict[str, Any]:
        return dict(self._stats)

    # ── компоновка кадра (решение Alex §1.6) ─────────────────────────────────
    def _compose_frame(self, i: int, trail: int, fig: Figure | None = None) -> Figure:
        p = self._p
        rec = self._history[i]
        if fig is None:
            fig = plt.figure(figsize=(16, 9))
        fig.clear()
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[2.6, 1.0],
                              hspace=0.3, wspace=0.18)
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        ax_field = fig.add_subplot(gs[0, 1])
        # ── 3D слева: детекции такта + истина
        if rec.points:
            xs = [pt.kx for pt in rec.points]; zs = [pt.ky for pt in rec.points]
            ys = [pt.pos for pt in rec.points]; cs = [pt.db for pt in rec.points]
            ax3d.scatter(xs, ys, zs, c=cs, cmap="turbo", vmin=-25, vmax=0, s=36, alpha=0.9)
        for name, (kx, ky, t0) in rec.truth.items():
            if name == "barrage":
                ax3d.plot([kx, kx], [0, p.n_axis], [ky, ky], "-",
                          color=_BARRAGE_COLOR, lw=4, alpha=0.5)
            else:
                ax3d.scatter([kx], [t0], [ky], marker="^",
                             color=_TRACK_COLOR if name == "target" else _COMB_COLOR, s=60)
        ax3d.set_xlim(-p.nx // 2, p.nx // 2); ax3d.set_zlim(-p.ny // 2, p.ny // 2)
        ax3d.set_ylim(0, p.n_axis)
        ax3d.set_xlabel("kx"); ax3d.set_ylabel("позиция по оси →"); ax3d.set_zlabel("ky")
        band = f" · полоса@{rec.band_angle}→null" if rec.banded else ""
        ax3d.set_title(f"такт {rec.tact}{band}", fontsize=10)
        ax3d.view_init(18, -60)
        # ── поле nx×ny справа: треки со следом K + номера
        # подписываем только устойчивые треки (>=3 тактов) — шумовые однодневки без №
        for tr in rec.tracks:
            hist = tr["history"][-trail:] if trail > 0 else tr["history"][-1:]
            n_h = len(hist)
            stable = len(tr["history"]) >= 3
            for j, (_t, hkx, hky, _r) in enumerate(hist):
                a = 0.15 + 0.85 * (j + 1) / n_h
                ax_field.plot(hkx, hky, "o", color=_TRACK_COLOR,
                              ms=(3 + 6 * a) if stable else 2.5, alpha=a if stable else 0.35)
            if stable:
                ax_field.annotate(f"№{tr['id']}" + ("✈" if tr["is_moving"] else ""),
                                  (tr["kx"] + 1.2, tr["ky"]), fontsize=9,
                                  color=_TRACK_COLOR, weight="bold")
        ckx, cky, _ = rec.truth["comb"]
        ax_field.plot(ckx, cky, "s", color=_COMB_COLOR, ms=9, fillstyle="none", mew=2)
        bkx, bky, _ = rec.truth["barrage"]
        ax_field.add_patch(plt.Rectangle((bkx - 1.5, bky - 1.5), 3, 3, fill=False,
                                         ec=_BARRAGE_COLOR, lw=2))
        ax_field.set_xlim(-p.nx // 2, p.nx // 2); ax_field.set_ylim(-p.ny // 2, p.ny // 2)
        ax_field.grid(alpha=0.25, color=_FG)
        ax_field.set_xlabel("kx (азимут)"); ax_field.set_ylabel("ky (угол места)")
        ax_field.set_title(f"поле {p.nx}×{p.ny}: треки, след {trail}", fontsize=10)
        # ── низ: ряд срезов по трекам + параметры
        n_sl = max(1, len(rec.slices))
        gs_low = gs[1, :].subgridspec(1, n_sl, wspace=0.25)
        for j in range(n_sl):
            axs = fig.add_subplot(gs_low[0, j])
            if j < len(rec.slices):
                sl = rec.slices[j]
                axs.imshow(sl["energy_db"].T, origin="lower", cmap="turbo",
                           vmin=-25, vmax=0, aspect="auto")
                ix = sl["kx"] + p.nx // 2; iy = sl["ky"] + p.ny // 2
                axs.add_patch(plt.Circle((ix, iy), 3.0, fill=False, ec="w", lw=1.6))
                feats = sl["features"]
                ftxt = " ".join(f"{k}={v:.2g}" for k, v in list(feats.items())[:6])
                axs.set_title(f"№{sl['track_id']} окно={sl['pos']} kx={sl['kx']:+.0f} "
                              f"ky={sl['ky']:+.0f} {'ЛЕТИТ' if sl['is_moving'] else '—'}",
                              fontsize=7)
                axs.set_xlabel(ftxt, fontsize=6)
            else:
                axs.axis("off")
        _apply_dark(fig)
        return fig

    def _fig_trajectory(self) -> Figure:
        """Вся траектория: истина vs трек (стиль p1_trajectory, тёмный)."""
        p = self._p
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        tx = [r.truth["target"][0] for r in self._history]
        ty = [r.truth["target"][1] for r in self._history]
        tz = [r.truth["target"][2] for r in self._history]
        ax.plot(tx, tz, ty, "--", color=_FG, lw=1.2, label="истина (цель)")
        cx = [r.truth["comb"][0] for r in self._history]
        cy = [r.truth["comb"][1] for r in self._history]
        cz = [r.truth["comb"][2] for r in self._history]
        ax.plot(cx, cz, cy, ":", color=_COMB_COLOR, lw=1.2, label="носитель гребёнки")
        # длиннейший трек
        if self._history[-1].tracks:
            tr = max(self._history[-1].tracks, key=lambda t: len(t["history"]))
            hx = [h[1] for h in tr["history"]]; hy = [h[2] for h in tr["history"]]
            hz = [h[3] * 32 for h in tr["history"]]
            ax.plot(hx, hz, hy, "-", color=_TRACK_COLOR, lw=2.2,
                    label=f"трек №{tr['id']}" + (" ✈" if tr["is_moving"] else ""))
        ax.set_xlabel("kx"); ax.set_ylabel("позиция по оси →"); ax.set_zlabel("ky")
        ax.legend(facecolor=_BG, edgecolor=_FG, labelcolor=_FG, fontsize=8)
        ax.set_title("Траектории за прогон: истина vs трек")
        ax.view_init(18, -60)
        _apply_dark(fig)
        return fig

    def _write_gifs_and_html(self) -> None:
        """GIF ×2 (с хвостом K / без) из ОДНОЙ истории (R5) + plotly HTML."""
        p = self._p
        out = Path("demo/graphics") / self.name
        out.mkdir(parents=True, exist_ok=True)
        gif_paths = {}
        for tag, trail in (("flight_trail", p.k_trail), ("flight_clean", 0)):
            fig = plt.figure(figsize=(16, 9))

            def _update(i: int, _trail=trail, _fig=fig):
                self._compose_frame(i, trail=_trail, fig=_fig)
                return []

            anim = FuncAnimation(fig, _update, frames=len(self._history),
                                 interval=200, blit=False)
            path = out / f"{tag}.gif"
            anim.save(str(path), writer=PillowWriter(fps=5))     # приём demo_body_motion.py:259
            plt.close(fig)
            gif_paths[tag] = str(path)
        self._stats["gifs"] = gif_paths
        try:                                                     # plotly — опционально
            import plotly.graph_objects as go
            from core.graphics.interactive import HtmlWriter
            tx = [r.truth["target"][0] for r in self._history]
            ty = [r.truth["target"][1] for r in self._history]
            tz = [r.truth["target"][2] for r in self._history]
            fig3d = go.Figure()
            fig3d.add_trace(go.Scatter3d(x=tx, y=tz, z=ty, mode="lines+markers",
                                         name="цель (истина)",
                                         line={"color": _TRACK_COLOR, "width": 5}))
            if self._history[-1].tracks:
                tr = max(self._history[-1].tracks, key=lambda t: len(t["history"]))
                fig3d.add_trace(go.Scatter3d(
                    x=[h[1] for h in tr["history"]], y=[h[3] * 32 for h in tr["history"]],
                    z=[h[2] for h in tr["history"]], mode="lines",
                    name=f"трек №{tr['id']}", line={"color": _COMB_COLOR, "width": 4}))
            fig3d.update_layout(template="plotly_dark",          # приём demo_body_motion.py:236
                                scene={"xaxis_title": "kx", "yaxis_title": "позиция",
                                       "zaxis_title": "ky"})
            self._stats["html"] = HtmlWriter(str(out)).write(fig3d, "flight_3d.html")
        except ImportError:
            self._stats["html"] = "plotly не установлен — HTML пропущен"


def main() -> None:
    report = Ex4Flight().run()
    print(report)
    for path in report.figures:
        print("  ", path)


if __name__ == "__main__":
    main()
