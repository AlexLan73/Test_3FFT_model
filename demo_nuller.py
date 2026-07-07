"""Демо SubspaceNuller: target + barrage под разными углами, сравнение до/после.

Сцена: цель kx=+2, barrage kx=-4 (разные половины углового поля).
SubspaceNuller (n_jammers=1) гасит barrage; цель под отличным углом выживает.

Запуск:
    python demo_nuller.py
Сохраняет:
    out/figures/nuller_before.png
    out/figures/nuller_after.png
"""
from __future__ import annotations

import numpy as np

from core.config import ArrayConfig, BarrageSpec, RangeConfig, SceneConfig, TargetSpec
from core.config.scene_config import ThermalNoiseSpec
from core.generators.scene import SceneBuilder, Synthesizer
from core.graphics import AngularMapVisualizer, FigureWriter
from core.models import AxisWindows, Fft3DModel, HannWindow
from core.models.anti_barrage import SubspaceNuller

# ── Параметры сцены ────────────────────────────────────────────────────────────

ARRAY = ArrayConfig(16, 16)
RNG   = RangeConfig(n_real=16, n_fft=64)

# Цель на правом луче (kx=+2), barrage с левого края (kx=-4) — ясно разнесены
SCENE_CFG = SceneConfig(
    emitters=(
        TargetSpec(kx=2, ky=0, range_bin=8, amplitude=1.0),
        BarrageSpec(kx=-4, ky=0, power=6.0),
    ),
    thermal=ThermalNoiseSpec(power=0.02),
)

# Для сравнения: barrage-only и target-only
BARRAGE_ONLY = SceneConfig(
    emitters=(BarrageSpec(kx=-4, ky=0, power=6.0),),
    thermal=ThermalNoiseSpec(power=0.02),
)
TARGET_ONLY = SceneConfig(
    emitters=(TargetSpec(kx=2, ky=0, range_bin=8, amplitude=1.0),),
    thermal=ThermalNoiseSpec(power=0.02),
)


def angular_energy_db(raw: np.ndarray, model: Fft3DModel) -> np.ndarray:
    """Угловая карта (nx, ny) в дБ, нормированная по максимуму."""
    cube = model.process(raw)
    return cube.angular_energy_db()


def peak_db_at(raw: np.ndarray, model: Fft3DModel, kx: int, ky: int) -> float:
    """Значение угловой карты в бине (kx, ky) [дБ, нормировка по своему максимуму]."""
    cube = model.process(raw)
    ix, iy = cube.index_of_angle(kx, ky)
    e = cube.angular_energy_db()
    return float(e[ix, iy])


def barrage_band_energy(raw: np.ndarray, model: Fft3DModel, kx_jam: int) -> float:
    """Суммарная энергия вблизи угла помехи (±2 бина по kx), нормировка по max."""
    cube = model.process(raw)
    ix, _ = cube.index_of_angle(kx_jam, 0)
    band = cube.angular_energy_db()[max(0, ix - 2):ix + 3, :]
    return float(band.max())


def main() -> None:
    builder   = SceneBuilder()
    synth     = Synthesizer(ARRAY, RNG, seed=42)
    model     = Fft3DModel(ARRAY, RNG, windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()))
    nuller    = SubspaceNuller(n_jammers=1)
    writer    = FigureWriter("./out/figures")
    vis       = AngularMapVisualizer(gate_kx=2.0, gate_ky=0.0, gate_half=1.5)

    # ── Синтез сцены target+barrage ──────────────────────────────────────────
    scene  = builder.build(SCENE_CFG)
    raw    = synth.build(scene)          # (16, 16, 16) complex128

    # ── Отчёт (до нуллинга) ──────────────────────────────────────────────────
    rep = nuller.report(raw)
    print("═" * 60)
    print("  SubspaceNuller · demo  (target kx=+2, barrage kx=-4)")
    print("═" * 60)
    print("  NullerReport (raw cube):")
    print(f"    λ₁/σ_n²   = {rep.lambda_ratio:.1f}")
    print(f"    occupancy = {rep.occupancy:.3f}")
    print(f"    is_barrage= {rep.is_barrage}")

    # ── Применение нуллера ───────────────────────────────────────────────────
    cleaned = nuller.apply(raw)          # (16, 16, 16) complex128, raw не изменён
    assert cleaned is not raw, "apply() должен вернуть новый массив!"
    assert raw.shape == cleaned.shape

    # ── Угловая карта ДО ─────────────────────────────────────────────────────
    cube_before = model.process(raw)
    fig_before  = vis.render(cube_before)
    writer.write(fig_before, "nuller_before.png")

    # ── Угловая карта ПОСЛЕ ──────────────────────────────────────────────────
    cube_after = model.process(cleaned)
    fig_after  = vis.render(cube_after)
    writer.write(fig_after, "nuller_after.png")

    # ── Метрики ──────────────────────────────────────────────────────────────
    # Подавление barrage (угловой бин kx=-4) в дБ
    e_b_before = barrage_band_energy(raw,     model, kx_jam=-4)
    e_b_after  = barrage_band_energy(cleaned, model, kx_jam=-4)
    suppression_db = e_b_before - e_b_after

    # Выживание цели (угловой бин kx=+2) в дБ (нормировка по своему полю)
    e_t_before = peak_db_at(raw,     model, kx=2, ky=0)
    e_t_after  = peak_db_at(cleaned, model, kx=2, ky=0)
    target_loss_db = e_t_before - e_t_after   # ≈ 0 = цель сохранена

    print()
    print("  Результаты нуллинга:")
    print(f"    barrage kx=-4: угл. энергия до={e_b_before:.1f} дБ, после={e_b_after:.1f} дБ")
    print(f"    → подавление помехи: {suppression_db:.1f} дБ")
    print()
    print(f"    цель kx=+2:    E до={e_t_before:.1f} дБ, после={e_t_after:.1f} дБ")
    print(f"    → потеря цели:  {target_loss_db:.1f} дБ  (0 = идеал, <3 дБ — хорошо)")
    print()

    # ── Отдельная проверка barrage-only и target-only ────────────────────────
    synth2 = Synthesizer(ARRAY, RNG, seed=7)

    raw_b = synth2.build(builder.build(BARRAGE_ONLY))
    rep_b = nuller.report(raw_b)
    print(f"  barrage-only сцена: λ_ratio={rep_b.lambda_ratio:.1f},"
          f" occ={rep_b.occupancy:.3f}, is_barrage={rep_b.is_barrage}")

    raw_t = synth2.build(builder.build(TARGET_ONLY))
    rep_t = nuller.report(raw_t)
    print(f"  target-only  сцена: λ_ratio={rep_t.lambda_ratio:.1f},"
          f" occ={rep_t.occupancy:.3f}, is_barrage={rep_t.is_barrage}")

    print()
    print("  PNG сохранены: out/figures/nuller_before.png, nuller_after.png")
    print("═" * 60)


if __name__ == "__main__":
    main()
