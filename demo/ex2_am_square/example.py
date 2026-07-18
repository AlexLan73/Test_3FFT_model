"""ex2 — сигналы ex1 (am+radio) в апертуре 64×64×4096, обнаружение объектов (гл.4-бис).

Канон — `MemoryBank/specs/demo_ex2_cube_2026-07-18.md` (§0 решения Alex, §3 сцена
6 объектов, §7 ревью R1-R8). Патент — `Doc/Patent/glava4bis_obyomnyj_primitiv.md`
(§4-бис.2а — адаптивная нарезка «грубо → всплеск → тонкий добор»).

Объём — комплексный IQ (эхо со steering-фазой по апертуре, реюз `render_pipeline`
ex1 через `WaveformFactory`), 6 объектов (2 типа × 3 длительности) кладутся БЕЗ шума
(`add_noise=False`), суммируются, шум добавляется ОДИН раз поверх суммы (R4).

Скан двухэтапный (патент §4-бис.2а):
  1. Грубо `AmToCube(depth=32, step=32)` — 128 окон без нахлёста. Порог — ДЕШЁВЫЙ
     векторизованный (пол шума Exp-статистикой по |куб|² на ПЕРВОМ кубе, реюз
     `estimate_noise_floor` из `demo/ex1_am_line/denoise.py`), НЕ токенизатор (R1:
     полный `VolumeTokenizer` на всех 128 грубых под-кубах — недопустимо медленно).
  2. Соседние всплески сливаются в ROI (R2), тонко `AmToCube(depth=16, step=8)` —
     ТОЛЬКО в ±1 грубом окне вокруг ROI, полный `VolumeTokenizer` — только здесь.

Источник данных (R8): `source="gen"|"disk"` — генерация сохраняет объём через
`DataContext.save_cube` и публикует в шину `"volume"`; disk — грузит и публикует
тем же ключом; канал единый (`Observer`), тракт обработки не знает источника.

Запуск:  .venv/Scripts/python.exe demo/ex2_am_square/example.py
         .venv/Scripts/python.exe demo/run_all.py --only ex2_am_square
Графики: demo/graphics/ex2_am_square/*.png  (в .gitignore)
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Чтобы работала форма `python demo/ex2_am_square/example.py` (конвенция репо).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from core.config import ArrayConfig, ProjectConfig  # noqa: E402
from core.data_context import DataContext  # noqa: E402
from core.generators.backends import NumpyBackend  # noqa: E402
from core.generators.waveforms import (  # noqa: E402
    AmToCube,
    Modulation,
    SignalField,
    TimeWindow,
    WaveformFactory,
    WaveformSpec,
)
from core.models.tokenizer import VolumeTokenizer  # noqa: E402
from demo.core import DemoContext, DemoReport, DemoRunner  # noqa: E402
from demo.ex1_am_line.denoise import estimate_noise_floor  # noqa: E402

KIND_AM, KIND_RADIO = "am", "radio"


# ── Value Objects (сцена) ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class ObjectSpec:
    """Один объект сцены (VO, §3 спеки): тип/длительность/позиция/угол."""

    name: str
    kind: str        # "am" | "radio"
    n_units: int       # периодов НЕСУЩЕЙ f_m (4/8/16)
    t0: int              # старт импульса, отсчёт по оси N=4096
    kx: int
    ky: int


_DEFAULT_SCENE: tuple[ObjectSpec, ...] = (
    ObjectSpec("A", KIND_AM, 4, 300, -20, 10),
    ObjectSpec("B", KIND_AM, 8, 1600, 0, 0),
    ObjectSpec("C", KIND_AM, 16, 2900, 15, -25),
    ObjectSpec("D", KIND_RADIO, 4, 900, 25, 20),
    ObjectSpec("E", KIND_RADIO, 8, 2200, -10, -15),
    ObjectSpec("F", KIND_RADIO, 16, 3500, -28, -5),
)


@dataclass(frozen=True)
class Ex2Params:
    """Все параметры примера (Value Object, требование Alex «всё переменное»)."""

    nx: int = 64
    ny: int = 64
    n_axis: int = 4096
    fs: float = 500e6
    f_m: float = 100e6
    m: float = 0.5
    env_frac: float = 1.0 / 8.0
    snr_db_list: tuple[float, ...] = (float("inf"), 10.0)   # из конфига (§0: 1.без шума 2.+10дБ)
    coarse_depth: int = 32
    coarse_step: int = 32
    fine_depth: int = 16
    fine_step: int = 8
    pfa: float = 1e-3
    seed: int = 7
    scene: tuple[ObjectSpec, ...] = field(default_factory=lambda: _DEFAULT_SCENE)
    angle_tol: int = 2       # допуск сопоставления по углу (бины, §4 спеки)
    # Динамический гейт (sidelobe blanking, как ex1-denoise): детекции слабее сильнейшей
    # на >= gate — боковики steering/окна, не отдельные объекты. Критично на clean-объёме:
    # без шума боковики решётки не замаскированы и токенизатор честно метит их 'source'
    # (диагностика ревью: 220 детекций, 176 вне допуска). Объекты равной амплитуды (1)
    # гейт не трогает.
    sidelobe_gate_db: float = 20.0


# ── реюз генераторов (не дублируем формулы, R3) ───────────────────────────────
def dur_samples(p: Ex2Params, n_units: int) -> int:
    """Длительность в отсчётах: n_units периодов НЕСУЩЕЙ f_m (как ex1 `dur_samples`)."""
    return int(round(n_units * p.fs / p.f_m))


def _object_spec(p: Ex2Params, obj: ObjectSpec) -> tuple[WaveformSpec, Modulation]:
    """`WaveformSpec` объекта (R3): meta {nx,ny,kx,ky} (+{m,f_m} для am), amplitude=1,
    `add_noise=False` -- 6 объектов суммируются ЧИСТЫМИ, шум добавляется один раз поверх суммы (R4)."""
    dur = dur_samples(p, obj.n_units)
    window = TimeWindow(kind="short", t0=obj.t0 / p.fs, dur=dur / p.fs)
    meta: dict[str, float] = {"nx": float(p.nx), "ny": float(p.ny),
                               "kx": float(obj.kx), "ky": float(obj.ky)}
    if obj.kind == KIND_AM:
        meta["m"] = p.m
        meta["f_m"] = p.f_m * p.env_frac
        modulation = Modulation.AM
    else:
        modulation = Modulation.CW
    spec = WaveformSpec(fs=p.fs, carrier_hz=p.f_m, n_samples=p.n_axis, amplitude=1.0,
                        window=window, meta=meta, add_noise=False)
    return spec, modulation


def build_clean_volume(p: Ex2Params, rng: np.random.Generator) -> np.ndarray:
    """Сумма 6 чистых полей (steering-эхо, R3/R6 патента: фаза до 3D-FFT, магнитуда после)."""
    factory = WaveformFactory()
    volume = np.zeros((p.nx, p.ny, p.n_axis), dtype=np.complex64)
    for obj in p.scene:
        spec, modulation = _object_spec(p, obj)
        field_: SignalField = factory.create(modulation).render(NumpyBackend(), spec, rng)
        volume = volume + field_.data
    return volume.astype(np.complex64)


def add_noise_volume(volume: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Шум ОДИН раз поверх суммы (R4) -- не мутирует `volume`, возвращает новый массив."""
    if not np.isfinite(snr_db):
        return volume
    noise_power = 1.0 / (10.0 ** (snr_db / 10.0))   # мощность сигнала=1 (amplitude=1)
    return NumpyBackend().add_noise(volume, noise_power, rng)


def _snr_tag(snr_db: float) -> str:
    return "clean" if not np.isfinite(snr_db) else f"snr{snr_db:+.0f}"


def _snr_label(snr_db: float) -> str:
    return "чистый (∞)" if not np.isfinite(snr_db) else f"SNR = {snr_db:+.0f} дБ"


# ── двухэтапный скан (патент §4-бис.2а, R1/R2) ────────────────────────────────
@dataclass(frozen=True)
class CoarsePoint:
    """Пик грубого окна над дешёвым порогом (VO, R1 -- НЕ токенизатор)."""

    kx: float
    ky: float
    pos: int
    db: float


def coarse_burst_points(volume: np.ndarray, cfg: ProjectConfig, p: Ex2Params) -> list[CoarsePoint]:
    """Грубый скан: дешёвый векторизованный порог по ПЕРВОМУ кубу (R1), без токенизатора.

    Пол шума N̂ = `estimate_noise_floor(|куб0|²)` (реюз `demo/ex1_am_line/denoise.py`);
    всплеск = max(|куб|²) > −N̂·ln(pfa) (Exp-статистика белого шума по мощности).
    """
    scanner = AmToCube(depth=p.coarse_depth, step=p.coarse_step)
    windows = scanner.scan(volume, cfg)
    if not windows:
        return []
    first_power = windows[0][1].magnitude.astype(np.float64).ravel() ** 2
    n_hat = estimate_noise_floor(first_power)
    threshold = -n_hat * math.log(p.pfa)

    points: list[CoarsePoint] = []
    for pos, cube in windows:
        power = cube.magnitude.astype(np.float64) ** 2
        peak_power = float(power.max())
        if peak_power <= threshold:
            continue
        idx = np.unravel_index(int(np.argmax(power)), power.shape)
        points.append(CoarsePoint(
            kx=float(cube.kx.values[idx[0]]), ky=float(cube.ky.values[idx[1]]),
            pos=pos, db=float(cube.magnitude_db[idx]),
        ))
    return points


def merge_rois(points: list[CoarsePoint], step: int, depth: int) -> list[tuple[int, int]]:
    """Слить соседние/смежные всплеск-окна в ROI (R2) -- иначе дубль-детекции у длинных импульсов."""
    if not points:
        return []
    positions = sorted({pt.pos for pt in points})
    rois: list[tuple[int, int]] = []
    cur_start = cur_end = positions[0]
    for pos in positions[1:]:
        if pos - cur_end <= step:            # смежное/соседнее окно -- расширяем текущий ROI
            cur_end = pos
        else:
            rois.append((cur_start, cur_end + depth))
            cur_start = cur_end = pos
    rois.append((cur_start, cur_end + depth))
    return rois


@dataclass(frozen=True)
class Detection:
    """Детекция тонкого добора (VO): угол/позиция окна/уровень/метка триажа."""

    kx: float
    ky: float
    window_pos: int
    window_depth: int
    level_db: float
    contrast_db: float
    label: str


def fine_scan_roi(volume: np.ndarray, cfg: ProjectConfig, p: Ex2Params,
                   roi: tuple[int, int]) -> list[Detection]:
    """Тонкий добор (патент §4-бис.2а) -- ТОЛЬКО в ±1 грубом окне вокруг ROI.

    Полный `VolumeTokenizer` -- ТОЛЬКО здесь (R1: на тонких под-кубах их единицы,
    не 128 как на грубом проходе). `window_l=fine_depth` -- окно токенизатора
    покрывает весь тонкий под-куб целиком (один токен на позицию скана, R6).
    """
    n = volume.shape[2]
    lo = max(0, roi[0] - p.coarse_step)
    hi = min(n, roi[1] + p.coarse_step)
    tokenizer = VolumeTokenizer(window_l=p.fine_depth)

    dets: list[Detection] = []
    pos = lo
    while pos < hi:
        sub = AmToCube(depth=p.fine_depth, step=p.fine_step, start=pos).fill(volume, cfg)
        tokens = tokenizer.tokenize(sub)
        mag_db = sub.magnitude_db
        median_db = float(np.median(mag_db))
        for tok in tokens:
            for peak in tok.peaks:
                ix, iy = sub.index_of_angle(peak.kx, peak.ky)
                peak_db = float(mag_db[ix, iy, :].max())
                dets.append(Detection(
                    kx=peak.kx, ky=peak.ky, window_pos=pos, window_depth=p.fine_depth,
                    level_db=peak_db, contrast_db=peak_db - median_db, label=tok.label,
                ))
        pos += p.fine_step
    return dets


def detect_objects(volume: np.ndarray, cfg: ProjectConfig, p: Ex2Params) -> list[Detection]:
    """Полный двухэтапный конвейер: грубо (дёшево) -> ROI (R2) -> тонко (полный токенизатор)
    -> динамический гейт `sidelobe_gate_db` (боковики steering слабее пика на >=gate — отброс)."""
    points = coarse_burst_points(volume, cfg, p)
    rois = merge_rois(points, p.coarse_step, p.coarse_depth)
    dets: list[Detection] = []
    for roi in rois:
        dets.extend(fine_scan_roi(volume, cfg, p, roi))
    if not dets:
        return dets
    gate = max(d.level_db for d in dets) - p.sidelobe_gate_db
    return [d for d in dets if d.level_db >= gate]


# ── метрики (§4 спеки) ─────────────────────────────────────────────────────────
def _overlaps(det: Detection, obj: ObjectSpec, p: Ex2Params) -> bool:
    obj_end = obj.t0 + dur_samples(p, obj.n_units)
    win_end = det.window_pos + det.window_depth
    angle_ok = abs(det.kx - obj.kx) <= p.angle_tol and abs(det.ky - obj.ky) <= p.angle_tol
    range_ok = det.window_pos < obj_end and win_end > obj.t0
    return angle_ok and range_ok


def match_metrics(dets: list[Detection], p: Ex2Params) -> dict[str, Any]:
    """found/missed/false + ошибка угла + контраст пика (§4 спеки).

    Матч: |Δkx|<=angle_tol, |Δky|<=angle_tol И окно накрывает импульс объекта.
    """
    found = 0
    angle_errs: list[float] = []
    contrasts: list[float] = []
    for obj in p.scene:
        candidates = [d for d in dets if _overlaps(d, obj, p)]
        if not candidates:
            continue
        best = max(candidates, key=lambda d: d.level_db)
        found += 1
        angle_errs.append(max(abs(best.kx - obj.kx), abs(best.ky - obj.ky)))
        contrasts.append(best.contrast_db)

    false = sum(1 for d in dets if not any(_overlaps(d, obj, p) for obj in p.scene))

    return {
        "found": found,
        "missed": len(p.scene) - found,
        "false": false,
        "angle_error_bins_mean": float(np.mean(angle_errs)) if angle_errs else None,
        "contrast_db_mean": float(np.mean(contrasts)) if contrasts else None,
    }


# ── DataContext-канал (R8) ─────────────────────────────────────────────────────
class _VolumeSink:
    """Наблюдатель шины `"volume"` (Observer): канал одинаков для gen|disk источника."""

    def __init__(self) -> None:
        self.volume: np.ndarray | None = None

    def on_data(self, key: str, data: object) -> None:
        if key == "volume":
            self.volume = data  # type: ignore[assignment]


# ── картинки (§5 спеки) ─────────────────────────────────────────────────────────
def fig_scene3d(points: list[CoarsePoint], p: Ex2Params, title: str) -> Figure:
    """3D scatter пиков грубого скана: оси (kx, ky, позиция окна), цвет=дБ; 6 объектов подписаны."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if points:
        xs = [pt.kx for pt in points]
        ys = [pt.ky for pt in points]
        zs = [pt.pos for pt in points]
        cs = [pt.db for pt in points]
        sc = ax.scatter(xs, ys, zs, c=cs, cmap="turbo", vmin=-25, vmax=0, s=40, alpha=0.85,
                        edgecolors="none")
        fig.colorbar(sc, ax=ax, shrink=0.7, label="дБ")
    for obj in p.scene:
        ax.scatter([obj.kx], [obj.ky], [obj.t0], marker="^", color="k", s=70)
        ax.text(obj.kx, obj.ky, obj.t0, f" {obj.name}:{obj.kind}/{obj.n_units}п", fontsize=7)
    ax.set_xlabel("kx (бины)")
    ax.set_ylabel("ky (бины)")
    ax.set_zlabel("позиция окна (отсчёт)")
    ax.set_title(title)
    ax.view_init(18, -60)
    fig.tight_layout()
    return fig


def fig_scene3d_noise_inset(points: list[CoarsePoint], p: Ex2Params, snr_db: float) -> Figure:
    """Врезки с шумом: 8-периодные объекты (B, E) при SNR из конфига, SNR в заголовке."""
    targets = [obj for obj in p.scene if obj.n_units == 8]
    fig = plt.figure(figsize=(4.5 * max(1, len(targets)), 4.2))
    for i, obj in enumerate(targets, start=1):
        ax = fig.add_subplot(1, len(targets), i, projection="3d")
        local = [pt for pt in points
                 if abs(pt.kx - obj.kx) <= 6 and abs(pt.ky - obj.ky) <= 6
                 and abs(pt.pos - obj.t0) <= 3 * p.coarse_step]
        if local:
            xs = [pt.kx for pt in local]
            ys = [pt.ky for pt in local]
            zs = [pt.pos for pt in local]
            cs = [pt.db for pt in local]
            ax.scatter(xs, ys, zs, c=cs, cmap="turbo", vmin=-25, vmax=0, s=30, edgecolors="none")
        ax.scatter([obj.kx], [obj.ky], [obj.t0], marker="^", color="k", s=60)
        ax.set_title(f"{obj.name} ({obj.kind}, 8 пер.)", fontsize=9)
        ax.set_xlabel("kx", fontsize=7)
        ax.set_ylabel("ky", fontsize=7)
        ax.set_zlabel("поз.окна", fontsize=7)
    fig.suptitle(f"С шумом — {_snr_label(snr_db)}")
    fig.tight_layout()
    return fig


def _aligned_pos(t0: int, step: int) -> int:
    return (t0 // step) * step


def fig_heatmaps(volume: np.ndarray, cfg: ProjectConfig, p: Ex2Params, snr_db: float) -> Figure:
    """Сетка угловых карт (реюз `cube.angular_energy_db()`, R7): окна вокруг объектов
    + пара пустых для контраста (~12 панелей)."""
    scanner = AmToCube(depth=p.coarse_depth, step=p.coarse_step)
    windows = scanner.scan(volume, cfg)
    cubes_by_pos = dict(windows)

    obj_positions = sorted({_aligned_pos(obj.t0, p.coarse_step) for obj in p.scene})
    chosen = [pos for pos in cubes_by_pos
              if any(abs(pos - op) <= p.coarse_step for op in obj_positions)]
    empties = [pos for pos in cubes_by_pos if pos not in chosen][:2]
    show_positions = sorted(set(chosen + empties))[:12]

    n = max(1, len(show_positions))
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for ax, pos in zip(flat_axes, show_positions, strict=False):
        cube = cubes_by_pos[pos]
        e = cube.angular_energy_db()
        im = ax.imshow(e.T, origin="lower", cmap="turbo", vmin=-25, vmax=0, aspect="equal")
        if pos in obj_positions:
            tag = " (окно объекта)"
        elif pos in chosen:
            tag = " (соседнее)"
        else:
            tag = " (пусто)"
        ax.set_title(f"поз={pos}{tag}", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.7)
    for ax in flat_axes[len(show_positions):]:
        ax.axis("off")
    fig.suptitle(f"Угловые карты (грубый скан, шаг {p.coarse_step}) · {_snr_label(snr_db)}")
    fig.tight_layout()
    return fig


# ── пример-обёртка (Template Method стенда) ─────────────────────────────────────
class Ex2AmSquare(DemoRunner):
    """АМ+radio в апертуре 64×64×4096, детекция гл.4-бис (двухэтапный скан)."""

    name = "ex2_am_square"

    def __init__(self, source: str = "gen", params: Ex2Params | None = None,
                 data_root: str = "./out/ex2_data") -> None:
        self._params = params or Ex2Params()
        self.seed = self._params.seed
        self._source = source
        self._dc = DataContext(root=data_root)
        self._sink = _VolumeSink()
        self._dc.subscribe("volume", self._sink)
        self._clean_volume: np.ndarray | None = None
        self._stats: dict[str, Any] = {}

    def _cfg(self) -> ProjectConfig:
        p = self._params
        return ProjectConfig(array=ArrayConfig(p.nx, p.ny), modulation="am",
                             am_window_depth=p.fine_depth, am_step=p.fine_step)

    def _volume_for_snr(self, ctx: DemoContext, clean_volume: np.ndarray, snr_db: float) -> np.ndarray:
        """Канал единый (R8): gen -- строит+сохраняет+публикует; disk -- грузит+публикует."""
        name = f"ex2_volume_{_snr_tag(snr_db)}"
        if self._source == "gen":
            vol = add_noise_volume(clean_volume, snr_db, ctx.rng)
            self._dc.save_cube(name, vol)
        else:
            vol = self._dc.load_cube(name)
        self._dc.publish("volume", vol)
        return self._sink.volume if self._sink.volume is not None else vol

    # ── hooks Template Method ───────────────────────────────────────────────
    def build_volume(self, ctx: DemoContext) -> np.ndarray | None:
        self._clean_volume = build_clean_volume(self._params, ctx.rng)
        return self._clean_volume

    def to_cube(self, ctx: DemoContext, volume: np.ndarray):
        p = self._params
        return AmToCube(depth=p.coarse_depth, step=p.coarse_step).fill(volume, self._cfg())

    def tokenize(self, ctx: DemoContext, cube):
        tok = VolumeTokenizer(window_l=cube.range.values.size)
        return tok.tokenize(cube), []

    def visualize(self, ctx: DemoContext) -> dict[str, Figure]:
        p = self._params
        cfg = self._cfg()
        clean_volume = self._clean_volume if self._clean_volume is not None \
            else build_clean_volume(p, ctx.rng)

        volumes: dict[float, np.ndarray] = {}
        points: dict[float, list[CoarsePoint]] = {}
        metrics: dict[str, Any] = {}
        for snr in p.snr_db_list:
            vol = self._volume_for_snr(ctx, clean_volume, snr)
            volumes[snr] = vol
            points[snr] = coarse_burst_points(vol, cfg, p)
            dets = detect_objects(vol, cfg, p)
            metrics[_snr_tag(snr)] = match_metrics(dets, p)

        figures: dict[str, Figure] = {}
        clean_snr = next((s for s in p.snr_db_list if not np.isfinite(s)), p.snr_db_list[0])
        figures["scene3d_clean"] = fig_scene3d(points[clean_snr], p, "3D-сцена БЕЗ шума")

        noisy_snr = next((s for s in p.snr_db_list if np.isfinite(s)), None)
        if noisy_snr is not None:
            figures["scene3d_noise_inset"] = fig_scene3d_noise_inset(points[noisy_snr], p, noisy_snr)

        for snr in p.snr_db_list:
            figures[f"heatmaps_{_snr_tag(snr)}"] = fig_heatmaps(volumes[snr], cfg, p, snr)

        self._stats = metrics
        return figures

    def report_metrics(self, ctx: DemoContext) -> dict[str, Any]:
        return dict(self._stats)


def main() -> None:
    report: DemoReport = Ex2AmSquare().run()
    print(report)
    for path in report.figures:
        print("  ", path)


if __name__ == "__main__":
    main()
