"""ex3 — правильные эхо-сигналы + ВСЕ 6 помех + подавление угла (база ex2).

Канон — `MemoryBank/specs/demo_ex3_echo_jammers_2026-07-18.md` (§0 решения Alex,
§8 ревью R1-R7). Три новых слоя над ex2:

1. **S1 — правильное эхо**: объекты строятся `build_pulse_echo_volume` (core) —
   задержанная копия зонда `s(t−t0)`, огибающая АМ с ФРОНТА импульса (не окно
   поверх глобальной волны, прецедент — `build_lfm_target_volume` для ЛЧМ).
2. **Помехи — все 6** (`jammers_rf`, реюз): по ОДНОЙ за прогон (решение Alex §0.2),
   JNR: barrage +20 дБ, остальные +10 дБ; своя угловая позиция (R7).
3. **Подавление угла помехи (R1)**: полосовая помеха (заград/CW/VFD) подсвечивает
   ВСЕ грубые окна на своём угле ⇒ эскалация ломается ⇒ конвейер жёсткий:
   грубый скан → карта полосы (угол = столбец всплесков) → гейт (тонкий добор НЕ
   пускать) → **null угла** → повторный скан → обычный тракт ex2.

   ⚠ Девиация от «MVDR-nuller» (§0.4) — задокументирована: `RobustMvdrNuller`/
   `SubspaceNuller` строят ковариацию M×M (M=nx·ny=4096 ⇒ 4096² complex128 =
   268 МБ + EVD/solve минуты) — дома непрактично на 64×64. Угол помехи ИЗВЕСТЕН
   из карты полосы, поэтому null = **rank-1 ортопроекция** по steering-вектору
   помехи: `y = x − a·(aᴴx)/(aᴴa)` — та же математика §4.1 (`SubspaceNuller.
   _ortho_projector` при E=[a/|a|]), но O(M·K) без матрицы M×M. В лёгких тестах
   (16×16, M=256) сверяется с полным `SubspaceNuller.apply` (EVD).

Запуск:  .venv/Scripts/python.exe demo/ex3_am_barrage/example.py
Графики: demo/graphics/ex3_am_barrage/*.png  (в .gitignore)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Конвенция репо: работает форма `python demo/ex3_am_barrage/example.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from core.generators.backends import NumpyBackend  # noqa: E402
from core.generators.grid import ArrayGrid  # noqa: E402
from core.generators.waveforms import (  # noqa: E402
    Modulation,
    TimeWindow,
    WaveformFactory,
    WaveformSpec,
    build_pulse_echo_volume,
)
from demo.core import DemoContext, DemoReport, DemoRunner  # noqa: E402
from demo.ex2_am_square.example import (  # noqa: E402  — база ex2, реюз (не копипаст)
    KIND_AM,
    CoarsePoint,
    Ex2Params,
    ObjectSpec,
    add_noise_volume,
    coarse_burst_points,
    detect_objects,
    dur_samples,
    fig_heatmaps,
    fig_scene3d,
    match_metrics,
)

# Ревью R7: углы помех не совпадают с объектами сцены.
_JAMMERS: tuple[tuple[str, Modulation, int, int], ...] = (
    ("barrage", Modulation.BARRAGE, 5, 18),
    ("drfm", Modulation.DRFM_REPEATER, -22, -20),
    ("smsp", Modulation.SMSP, 10, -8),
    ("industrial_cw", Modulation.INDUSTRIAL_CW, -5, 25),
    ("impulsive_arc", Modulation.IMPULSIVE_ARC, 0, -28),
    ("vfd", Modulation.VFD_HARMONIC, 28, -12),
)


@dataclass(frozen=True)
class Ex3Params:
    """Параметры ex3: база ex2 (композиция) + помехи (Value Object)."""

    base: Ex2Params = field(default_factory=Ex2Params)
    jnr_barrage_db: float = 20.0
    jnr_other_db: float = 10.0
    snr_db: float = 10.0                 # шумовой прогон (clean — отдельно, отладка §0.3 ex2)
    jammers: tuple[tuple[str, Modulation, int, int], ...] = field(default_factory=lambda: _JAMMERS)
    band_gate_frac: float = 1.0 / 3.0    # R1: всплесков больше трети окон = «полоса»
    drfm_lead0: int | None = None        # ex4: позиция 1-й копии гребёнки (None = 0.15·N)


# ── генерация (S1: правильные эхо; помехи из реестра) ────────────────────────
def build_echo_volume(p: Ex3Params, rng: np.random.Generator) -> np.ndarray:
    """Сумма 6 ПРАВИЛЬНЫХ эх (`build_pulse_echo_volume`, огибающая с фронта), без шума."""
    b = p.base
    volume = np.zeros((b.nx, b.ny, b.n_axis), dtype=np.complex64)
    for obj in b.scene:
        extra = {"m": b.m, "f_m": b.f_m * b.env_frac} if obj.kind == KIND_AM else None
        volume = volume + build_pulse_echo_volume(
            Modulation.AM if obj.kind == KIND_AM else Modulation.CW,
            fs=b.fs, carrier_hz=b.f_m, n_samples=b.n_axis,
            dur_samples=dur_samples(b, obj.n_units), t0_samples=obj.t0,
            kx=obj.kx, ky=obj.ky, nx=b.nx, ny=b.ny, rng=rng, extra_meta=extra,
        )
    return volume.astype(np.complex64)


def build_drfm_comb_volume(p: Ex3Params, kx: int, ky: int,
                           rng: np.random.Generator) -> np.ndarray:
    """DRFM-ГРЕБЁНКА для импульсного АМ-тракта: серия задержанных копий НАШЕГО зонда.

    ⚠ Почему не `DrfmRepeaterJammer` (core): тот ретранслирует ЛЧМ-чирп во всю ось
    (гребёнка из него получается в ЛЧМ-тракте гл.3 ПОСЛЕ дечирпа); в АМ-тракте ex3
    (без дечирпа) полный чирп выглядит сплошной полосой. Физика DRFM: ретранслятор
    повторяет то, что УСЛЫШАЛ — наш импульсный зонд ⇒ ложные цели = `count` коротких
    эхо-копий импульса (реюз `build_pulse_echo_volume`, S1) с шагом `spacing` и
    затуханием `decay` по копиям — в кубе цепочка пиков = comb (патент §4-бис.4/гл.4).
    """
    b = p.base
    amp0 = 10.0 ** (p.jnr_other_db / 20.0)
    # доли оси — масштабируются на лёгкие тестовые размеры (512) и полный 4096;
    # lead0 (ex4) — позиция первой копии от НОСИТЕЛЯ гребёнки (движется по тактам)
    lead = p.drfm_lead0 if p.drfm_lead0 is not None else round(0.15 * b.n_axis)
    spacing = round(0.10 * b.n_axis)
    count, decay = 5, 0.85
    n_units = 8                               # ретранслируется 8-периодный зонд (как B/E)
    volume = np.zeros((b.nx, b.ny, b.n_axis), dtype=np.complex64)
    for i in range(count):
        t0 = lead + i * spacing
        if t0 >= b.n_axis:
            break
        volume = volume + build_pulse_echo_volume(
            Modulation.AM, fs=b.fs, carrier_hz=b.f_m, n_samples=b.n_axis,
            dur_samples=dur_samples(b, n_units), t0_samples=t0,
            kx=kx, ky=ky, nx=b.nx, ny=b.ny, rng=rng,
            amplitude=amp0 * decay ** i,
            extra_meta={"m": b.m, "f_m": b.f_m * b.env_frac},
        )
    return volume.astype(np.complex64)


def build_jammer_volume(p: Ex3Params, name: str, modulation: Modulation,
                        kx: int, ky: int, rng: np.random.Generator) -> np.ndarray:
    """Помеха из реестра `WaveformFactory` (реюз, R5): amplitude=10^(JNR/20), свой угол.

    `drfm` — особый путь (`build_drfm_comb_volume`): гребёнка коротких эхо-копий
    зонда, а не ЛЧМ-ретранслятор core (см. докстринг там).
    """
    if name == "drfm":
        return build_drfm_comb_volume(p, kx, ky, rng)
    b = p.base
    jnr = p.jnr_barrage_db if name == "barrage" else p.jnr_other_db
    spec = WaveformSpec(
        fs=b.fs, carrier_hz=b.f_m, n_samples=b.n_axis,
        amplitude=10.0 ** (jnr / 20.0),
        window=TimeWindow(kind="full"),
        meta={"nx": float(b.nx), "ny": float(b.ny), "kx": float(kx), "ky": float(ky)},
        add_noise=False,
    )
    return WaveformFactory().create(modulation).render(NumpyBackend(), spec, rng).data


# ── R1: карта полосы + rank-1 null угла ──────────────────────────────────────
def band_angle(points: list[CoarsePoint], n_windows: int,
               gate_frac: float) -> tuple[float, float] | None:
    """Угол «полосы»: (kx,ky), всплывший в > gate_frac окон. None — полосы нет.

    Дешёвая сборка по окнам (без токенизатора, R1): группировка всплесков по углу.
    """
    counts: dict[tuple[float, float], int] = {}
    for pt in points:
        key = (pt.kx, pt.ky)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    angle, n_hits = max(counts.items(), key=lambda kv: kv[1])
    return angle if n_hits > gate_frac * n_windows else None


def null_angle(volume: np.ndarray, kx: float, ky: float) -> np.ndarray:
    """Rank-1 ортопроекция: подавить известный угол (kx,ky) — `y = x − a·(aᴴx)/(aᴴa)`.

    Математика = `SubspaceNuller` §4.1 (P⊥ = I − EEᴴ) при E=[a/‖a‖] (угол помехи
    известен из карты полосы — EVD не нужен); O(M·K) без матрицы M×M (докстринг
    модуля, девиация от MVDR по памяти/времени на 64×64). Вход не мутируется.
    """
    nx, ny, k_snap = volume.shape
    a = ArrayGrid(nx, ny).steering(kx, ky).ravel().astype(np.complex128)
    x_mat = volume.reshape(nx * ny, k_snap)
    coeffs = (a.conj() @ x_mat) / float(np.real(a.conj() @ a))     # (K,)
    cleaned = x_mat - np.outer(a, coeffs)
    return cleaned.reshape(nx, ny, k_snap).astype(volume.dtype)


# ── пример-обёртка ───────────────────────────────────────────────────────────
class Ex3AmBarrage(DemoRunner):
    """Эхо-сцена ex2 + одна помеха за прогон + null полосы (патент §4-бис.2а / 4-бис.4)."""

    name = "ex3_am_barrage"

    def __init__(self, params: Ex3Params | None = None) -> None:
        self._p = params or Ex3Params()
        self.seed = self._p.base.seed
        self._stats: dict[str, Any] = {}

    def _cfg(self):
        # реюз конфига ex2 (тот же тракт AmToCube)
        from demo.ex2_am_square.example import Ex2AmSquare
        return Ex2AmSquare(params=self._p.base)._cfg()

    def _run_pipeline(self, volume: np.ndarray, cfg) -> tuple[dict[str, Any], list[CoarsePoint],
                                                              list[CoarsePoint] | None, bool, list]:
        """Конвейер R1: скан → полоса? → null → повторный скан → тонкий тракт ex2.

        Возвращает (метрики, точки ДО, точки ПОСЛЕ null|None, полоса_была, детекции) —
        детекции 5-м элементом нужны ex4 (трекинг по тактам).
        """
        p = self._p
        b = p.base
        n_windows = b.n_axis // b.coarse_step
        pts_before = coarse_burst_points(volume, cfg, b)
        angle = band_angle(pts_before, n_windows, p.band_gate_frac)
        pts_after: list[CoarsePoint] | None = None
        work = volume
        if angle is not None:                       # полоса: гейт → null → повторный скан
            work = null_angle(volume, *angle)
            pts_after = coarse_burst_points(work, cfg, b)
        dets = detect_objects(work, cfg, b)          # тонкий тракт ex2 (полный токенизатор)
        metrics = match_metrics(dets, b)
        metrics["band_detected"] = angle is not None
        metrics["band_angle"] = angle
        return metrics, pts_before, pts_after, angle is not None, dets

    def visualize(self, ctx: DemoContext) -> dict[str, Figure]:
        p = self._p
        b = p.base
        cfg = self._cfg()
        figures: dict[str, Figure] = {}
        stats: dict[str, Any] = {}

        echo_clean = build_echo_volume(p, ctx.rng)

        # базовые прогоны: clean (отладка) и snr+10 — без помех
        for tag, snr in (("clean", float("inf")), ("snr10", p.snr_db)):
            vol = add_noise_volume(echo_clean, snr, ctx.rng)
            metrics, pts, _, _, _ = self._run_pipeline(vol, cfg)
            stats[f"base_{tag}"] = metrics
            if tag == "clean":
                figures["scene3d_signals"] = fig_scene3d(pts, b, "Правильные эхо (S1), без помех")

        # помехи: по одной за прогон (решение Alex), SNR фиксирован
        for name, modulation, jkx, jky in p.jammers:
            jam = build_jammer_volume(p, name, modulation, jkx, jky, ctx.rng)
            vol = add_noise_volume(echo_clean + jam, p.snr_db, ctx.rng)
            metrics, pts_before, pts_after, banded, _ = self._run_pipeline(vol, cfg)
            stats[name] = metrics
            figures[f"{name}_before"] = fig_scene3d(
                pts_before, b, f"{name}: грубый скан ДО подавления (JNR "
                f"{p.jnr_barrage_db if name == 'barrage' else p.jnr_other_db:+.0f} дБ)")
            if pts_after is not None:
                figures[f"{name}_after"] = fig_scene3d(
                    pts_after, b, f"{name}: ПОСЛЕ null угла {metrics['band_angle']}")
            if name == "barrage":
                figures["heatmaps_barrage"] = fig_heatmaps(vol, cfg, b, p.snr_db)

        self._stats = stats
        return figures

    def report_metrics(self, ctx: DemoContext) -> dict[str, Any]:
        return {
            run: (f"found {m['found']}/6 false {m['false']}"
                  + (f" · полоса@{m['band_angle']}→null" if m.get("band_detected") else ""))
            for run, m in self._stats.items()
        }


def main() -> None:
    report: DemoReport = Ex3AmBarrage().run()
    print(report)
    for path in report.figures:
        print("  ", path)


if __name__ == "__main__":
    main()
