"""ex5 — параболическое уточнение пика В КУБЕ (суб-биновая точность, TASK_ex1_search_p2 §3).

Идея показа: 3 цели ставятся на ДРОБНЫЕ угловые бины (steering `ArrayGrid` — честный
непрерывный фазовый вектор) и ДРОБНЫЙ частотный бин (несущая не кратна fs/D) —
т.е. источники сидят МЕЖДУ ячейками куба, как в жизни. Целочисленный argmax ошибается
до ±0.5 бина; лог-парабола по 3 точкам вдоль каждой оси (`core.models.tokenizer.
refine_peak`) восстанавливает дробную позицию.

Конвейер: сцена (WaveformFactory, CW-эхо со steering) → сумма + шум (реюз ex2)
→ AmToCube (3D-FFT, Хэмминг×Хэмминг×Ханн) → топ-3 пика (NMS) → refine_peak
→ сравнение [argmax | парабола] с истиной по всем 3 осям (kx, ky, range).

Запуск:  .venv/Scripts/python.exe demo/ex5_peak_refine/example.py
         .venv/Scripts/python.exe demo/run_all.py --only ex5_peak_refine
Графики: demo/graphics/ex5_peak_refine/*.png  (в .gitignore)
  cuts_<snr>.png — срезы куба через пик по каждой оси: точки, парабола, истина/argmax/уточнение
  map_<snr>.png  — угловая карта энергии + зум 7×7 бинов вокруг каждой цели
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Чтобы работала форма `python demo/ex5_peak_refine/example.py` (конвенция репо).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from core.config import ArrayConfig, ProjectConfig  # noqa: E402
from core.generators.backends import NumpyBackend  # noqa: E402
from core.generators.waveforms import (  # noqa: E402
    AmToCube,
    Modulation,
    TimeWindow,
    WaveformFactory,
    WaveformSpec,
)
from core.models.result import SpectralCube  # noqa: E402
from core.models.tokenizer import RefinedPeak, refine_peak  # noqa: E402
from demo.core import DemoContext, DemoRunner  # noqa: E402
from demo.ex2_am_square.example import add_noise_volume  # noqa: E402

AXIS_NAMES = ("kx", "ky", "range")


# ── Value Objects ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FracObject:
    """Точечный источник на ДРОБНОЙ позиции куба (VO): угловые бины + несущая."""

    name: str
    kx: float           # дробный угловой бин X (между ячейками)
    ky: float           # дробный угловой бин Y
    freq_hz: float      # несущая; дробный частотный бин = freq_hz·D/fs


_DEFAULT_SCENE: tuple[FracObject, ...] = (
    FracObject("A", kx=-8.30, ky=5.60, freq_hz=100.30e6),
    FracObject("B", kx=0.45, ky=-10.25, freq_hz=79.70e6),
    FracObject("C", kx=11.70, ky=-3.85, freq_hz=121.55e6),
)


@dataclass(frozen=True)
class Ex5Params:
    """Все параметры примера (VO, «всё переменное» — конвенция demo-серии)."""

    nx: int = 32
    ny: int = 32
    n_axis: int = 256
    depth: int = 256                                    # окно 3D-FFT = вся ось (один куб)
    fs: float = 500e6
    snr_db_list: tuple[float, ...] = (float("inf"), 0.0)
    seed: int = 7
    scene: tuple[FracObject, ...] = field(default_factory=lambda: _DEFAULT_SCENE)
    guard: tuple[int, int, int] = (3, 3, 6)             # NMS-блок вокруг найденного пика
    cut_half: int = 5                                   # полуширина среза на графике (бины)
    zoom_half: int = 3                                  # полуширина зума угловой карты


# ── сцена (реюз генераторов, формулы не дублируем) ────────────────────────────
def true_index(p: Ex5Params, obj: FracObject) -> tuple[float, float, float]:
    """Истинный ДРОБНЫЙ индекс объекта в кубе: (kx+nx/2, ky+ny/2, f·D/fs)."""
    return (obj.kx + p.nx / 2.0, obj.ky + p.ny / 2.0, obj.freq_hz * p.depth / p.fs)


def build_clean_volume(p: Ex5Params, rng: np.random.Generator) -> np.ndarray:
    """Сумма CW-эхо со steering по дробным (kx, ky) — реюз `WaveformFactory` (как ex2)."""
    factory = WaveformFactory()
    volume = np.zeros((p.nx, p.ny, p.n_axis), dtype=np.complex64)
    for obj in p.scene:
        window = TimeWindow(kind="short", t0=0.0, dur=p.n_axis / p.fs)
        meta = {"nx": float(p.nx), "ny": float(p.ny), "kx": obj.kx, "ky": obj.ky}
        spec = WaveformSpec(fs=p.fs, carrier_hz=obj.freq_hz, n_samples=p.n_axis,
                            amplitude=1.0, window=window, meta=meta, add_noise=False)
        volume = volume + factory.create(Modulation.CW).render(NumpyBackend(), spec, rng).data
    return volume.astype(np.complex64)


# ── детекция + уточнение ──────────────────────────────────────────────────────
def top_peaks(power: np.ndarray, n_peaks: int,
              guard: tuple[int, int, int]) -> list[tuple[int, ...]]:
    """До `n_peaks` максимумов с NMS-занулением guard-блока (стиль `coarse_burst_points`)."""
    work = power.astype(np.float64).copy()
    peaks: list[tuple[int, ...]] = []
    for _ in range(n_peaks):
        idx = tuple(int(v) for v in np.unravel_index(int(np.argmax(work)), work.shape))
        if not np.isfinite(work[idx]) or work[idx] <= 0.0:
            break
        peaks.append(idx)
        sl = tuple(slice(max(0, i - g), min(s, i + g + 1))
                   for i, g, s in zip(idx, guard, work.shape, strict=True))
        work[sl] = -np.inf
    return peaks


@dataclass(frozen=True)
class RefineResult:
    """Итог по одному объекту (VO): истина, argmax и парабола + ошибки по осям."""

    obj: FracObject
    truth: tuple[float, float, float]
    peak: RefinedPeak

    @property
    def err_argmax(self) -> tuple[float, ...]:
        return tuple(abs(i - t) for i, t in zip(self.peak.index, self.truth, strict=True))

    @property
    def err_refined(self) -> tuple[float, ...]:
        return tuple(abs(f - t) for f, t in zip(self.peak.frac_index, self.truth, strict=True))


def analyze_cube(cube: SpectralCube, p: Ex5Params) -> list[RefineResult]:
    """Топ-N пиков куба → парабола → сопоставление с истиной по близости угла."""
    power = cube.magnitude.astype(np.float64) ** 2
    peaks = top_peaks(power, len(p.scene), p.guard)
    results: list[RefineResult] = []
    used: set[int] = set()
    for obj in p.scene:
        t = true_index(p, obj)
        best_k, best_d = -1, float("inf")
        for k, idx in enumerate(peaks):
            if k in used:
                continue
            d = (idx[0] - t[0]) ** 2 + (idx[1] - t[1]) ** 2
            if d < best_d:
                best_k, best_d = k, d
        if best_k < 0:
            continue
        used.add(best_k)
        results.append(RefineResult(obj=obj, truth=t, peak=refine_peak(power, peaks[best_k])))
    return results


# ── пример (Template Method) ──────────────────────────────────────────────────
class Ex5PeakRefine(DemoRunner):
    """Парабола в кубе: [сцена на дробных бинах → AmToCube → топ-3 NMS → refine_peak]."""

    name = "ex5_peak_refine"
    seed = 7

    def __init__(self, params: Ex5Params | None = None) -> None:
        self._p = params if params is not None else Ex5Params()
        self.seed = self._p.seed
        self._stats: dict[str, Any] = {}

    def _cfg(self) -> ProjectConfig:
        return ProjectConfig(array=ArrayConfig(self._p.nx, self._p.ny), modulation="am")

    def visualize(self, ctx: DemoContext) -> dict[str, Figure]:
        p = self._p
        clean = build_clean_volume(p, ctx.rng)
        scanner = AmToCube(depth=p.depth, step=64)
        figures: dict[str, Figure] = {}
        for snr in p.snr_db_list:
            tag = "clean" if not np.isfinite(snr) else f"snr{snr:+.0f}"
            volume = add_noise_volume(clean, snr, ctx.rng)
            cube = scanner.fill(volume, self._cfg())
            results = analyze_cube(cube, p)
            figures[f"cuts_{tag}"] = self._fig_cuts(cube, results, snr)
            figures[f"map_{tag}"] = self._fig_map(cube, results, snr)
            self._collect(tag, results)
        return figures

    def report_metrics(self, ctx: DemoContext) -> dict[str, Any]:
        return dict(self._stats)

    def _collect(self, tag: str, results: list[RefineResult]) -> None:
        e_arg = [e for r in results for e in r.err_argmax]
        e_ref = [e for r in results for e in r.err_refined]
        mean_arg, mean_ref = float(np.mean(e_arg)), float(np.mean(e_ref))
        self._stats[tag] = (f"объектов {len(results)}/{len(self._p.scene)} · "
                            f"ошибка argmax {mean_arg:.3f} бина → парабола {mean_ref:.3f} "
                            f"(×{mean_arg / max(mean_ref, 1e-9):.1f} точнее)")

    # ── графика (внутри примера, конвенция demo-серии ex1–ex4) ────────────────
    def _fig_cuts(self, cube: SpectralCube, results: list[RefineResult], snr: float) -> Figure:
        p = self._p
        power = cube.magnitude.astype(np.float64) ** 2
        n_rows = len(results)
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.1 * n_rows))
        axes = np.atleast_2d(axes)
        for row, r in enumerate(results):
            for ax_i in range(3):
                self._panel_cut(axes[row, ax_i], power, r, ax_i)
        snr_tag = "чистый (∞)" if not np.isfinite(snr) else f"SNR={snr:+.0f} дБ"
        fig.suptitle(f"ex5 · срезы куба {p.nx}×{p.ny}×{p.depth} через пик · {snr_tag} · "
                     "точки=бины (дБ), кривая=лог-парабола по 3 точкам; "
                     "зелёная=истина, серая=argmax, оранжевая=парабола", y=0.995)
        fig.tight_layout()
        return fig

    def _panel_cut(self, ax: Any, power: np.ndarray, r: RefineResult, ax_i: int) -> None:
        i0 = r.peak.index[ax_i]
        n = power.shape[ax_i]
        lo, hi = max(0, i0 - self._p.cut_half), min(n, i0 + self._p.cut_half + 1)
        sel = list(r.peak.index)
        sel[ax_i] = slice(lo, hi)                       # 1D-срез через пик вдоль оси ax_i
        line = power[tuple(sel)]
        ref_db = 10.0 * np.log10(max(float(power[r.peak.index]), 1e-30))
        bins = np.arange(lo, hi)
        ax.plot(bins, 10.0 * np.log10(np.maximum(line, 1e-30)) - ref_db, "o",
                color="tab:blue", ms=4)

        # лог-парабола через 3 центральные точки — та же, что внутри refine_peak
        if 0 < i0 < n - 1:
            l3 = np.log10(np.maximum(power[tuple(
                sel[:ax_i] + [slice(i0 - 1, i0 + 2)] + sel[ax_i + 1:])], 1e-30))
            a = 0.5 * (l3[2] - 2.0 * l3[1] + l3[0])
            b = 0.5 * (l3[2] - l3[0])
            d = np.linspace(-1.4, 1.4, 120)
            ax.plot(i0 + d, 10.0 * (l3[1] + b * d + a * d * d) - ref_db,
                    "-", color="tab:orange", lw=1.2, alpha=0.8)

        truth, frac = r.truth[ax_i], r.peak.frac_index[ax_i]
        ax.axvline(truth, color="tab:green", lw=1.6, label="истина")
        ax.axvline(i0, color="gray", lw=1.2, ls="--", label="argmax")
        ax.axvline(frac, color="tab:orange", lw=1.4, ls=":", label="парабола")
        ax.set_title(f"{r.obj.name} · {AXIS_NAMES[ax_i]} · истина {truth:.2f} · "
                     f"argmax {i0} (ошибка {r.err_argmax[ax_i]:.2f})\n"
                     f"парабола {frac:.2f} (ошибка {r.err_refined[ax_i]:.3f})", fontsize=7.5)
        ax.set_xlabel("бин", fontsize=7)
        ax.set_ylabel("дБ отн. пика", fontsize=7)
        ax.grid(alpha=0.3)
        if ax_i == 0:
            ax.legend(fontsize=6, loc="lower left")

    def _fig_map(self, cube: SpectralCube, results: list[RefineResult], snr: float) -> Figure:
        p = self._p
        e_db = cube.angular_energy_db()
        n_zoom = len(results)
        fig, axes = plt.subplots(1, 1 + n_zoom, figsize=(4.6 * (1 + n_zoom), 4.4))
        extent = (float(cube.ky.values[0]) - 0.5, float(cube.ky.values[-1]) + 0.5,
                  float(cube.kx.values[0]) - 0.5, float(cube.kx.values[-1]) + 0.5)
        im = axes[0].imshow(e_db, origin="lower", extent=extent, aspect="auto",
                            cmap="viridis", vmin=-40.0, vmax=0.0)
        axes[0].set_title(f"угловая энергия (дБ), {p.nx}×{p.ny}", fontsize=8)
        axes[0].set_xlabel("ky (бин)"), axes[0].set_ylabel("kx (бин)")
        fig.colorbar(im, ax=axes[0], shrink=0.85)
        for r in results:
            axes[0].plot(r.obj.ky, r.obj.kx, "+", color="lime", ms=10, mew=1.6)

        half = p.zoom_half
        for k, r in enumerate(results):
            ax = axes[1 + k]
            ix, iy = r.peak.index[0], r.peak.index[1]
            x0, x1 = max(0, ix - half), min(p.nx, ix + half + 1)
            y0, y1 = max(0, iy - half), min(p.ny, iy + half + 1)
            sub = e_db[x0:x1, y0:y1]
            ext = (float(cube.ky.values[y0]) - 0.5, float(cube.ky.values[y1 - 1]) + 0.5,
                   float(cube.kx.values[x0]) - 0.5, float(cube.kx.values[x1 - 1]) + 0.5)
            ax.imshow(sub, origin="lower", extent=ext, aspect="auto", cmap="viridis")
            ax.plot(r.obj.ky, r.obj.kx, "+", color="lime", ms=14, mew=2.0, label="истина")
            ax.plot(cube.ky.values[iy], cube.kx.values[ix], "x", color="w",
                    ms=10, mew=1.6, label="argmax")
            fx, fy = r.peak.frac_index[0] - p.nx / 2.0, r.peak.frac_index[1] - p.ny / 2.0
            ax.plot(fy, fx, "o", mfc="none", mec="tab:orange", ms=11, mew=2.0, label="парабола")
            da = float(np.hypot(*r.err_argmax[:2]))
            dr = float(np.hypot(*r.err_refined[:2]))
            ax.set_title(f"зум {r.obj.name}: угловая ошибка argmax {da:.2f} → "
                         f"парабола {dr:.3f} бина", fontsize=8)
            ax.set_xlabel("ky (бин)", fontsize=7)
            if k == 0:
                ax.legend(fontsize=6, loc="lower right")
        snr_tag = "чистый (∞)" if not np.isfinite(snr) else f"SNR={snr:+.0f} дБ"
        fig.suptitle(f"ex5 · пик между бинами куба · {snr_tag} · "
                     "лайм=истина (дробный угол), белый=argmax (целый бин), "
                     "оранжевый=парабола", y=0.99)
        fig.tight_layout()
        return fig


def main() -> None:
    report = Ex5PeakRefine().run()
    print(report)
    for path in report.figures:
        print("  ", path)


if __name__ == "__main__":
    main()
