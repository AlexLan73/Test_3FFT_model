"""Демо CA-CFAR детектора: target + barrage под разными углами, сравнение до/после нуллинга.

Сцена: цель kx=+2, range_bin=8, barrage kx=-4 (ясно разнесены по углу).
SubspaceNuller гасит barrage; CaCfarDetector находит цель и не видит заливку.

Запуск:
    python demo_cfar.py
    python demo_cfar.py --nx 6 --ny 15   # неквадратная апертура i×j (E5), паддинг до 2ⁿ
"""
from __future__ import annotations

import argparse

import numpy as np

from core.config import ArrayConfig, BarrageSpec, RangeConfig, SceneConfig, TargetSpec
from core.config.scene_config import ThermalNoiseSpec
from core.generators.scene import SceneBuilder, Synthesizer
from core.models import AxisWindows, Fft3DModel, HannWindow
from core.models.anti_barrage import CaCfarDetector, Detection, SubspaceNuller

# ── Параметры сцены ────────────────────────────────────────────────────────────

RNG = RangeConfig(n_real=16, n_fft=64)

TARGET_KX    = 2
TARGET_KY    = 0
TARGET_RANGE = 8    # ожидаемый бин дальности после 3D-БПФ

BARRAGE_KX = -4
BARRAGE_KY = 0

SCENE_CFG = SceneConfig(
    emitters=(
        TargetSpec(kx=TARGET_KX, ky=TARGET_KY, range_bin=TARGET_RANGE, amplitude=1.0),
        BarrageSpec(kx=BARRAGE_KX, ky=BARRAGE_KY, power=6.0),
    ),
    thermal=ThermalNoiseSpec(power=0.02),
)


def _print_detections(
    detections: list[Detection],
    label: str,
    target_kx: int,
    target_range: int,
    barrage_kx: int,
) -> None:
    print(f"\n  Детекции ({label}): {len(detections)} шт.")
    target_found = False
    barrage_fa = 0
    for d in detections:
        marker = ""
        if d.kx == target_kx and abs(d.range_bin - target_range) <= 1:
            marker = "  ← ЦЕЛЬ ✅"
            target_found = True
        elif d.kx == barrage_kx:
            barrage_fa += 1
            marker = "  ← ЛОЖНАЯ (barrage column) ⚠️"
        print(
            f"    kx={d.kx:+.0f} ky={d.ky:+.0f}"
            f"  range={d.range_bin:3d}"
            f"  level={d.level_db:+6.1f} дБ"
            f"  thr={d.threshold_db:+6.1f} дБ"
            f"{marker}"
        )
    if not detections:
        print("    (нет детекций)")
    print(f"  → Цель найдена: {target_found}  |  ложных в столбе barrage: {barrage_fa}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Демо CA-CFAR детектора (§E5: апертура i×j).")
    parser.add_argument("--nx", type=int, default=16, help="число элементов решётки по X (дефолт 16)")
    parser.add_argument("--ny", type=int, default=16, help="число элементов решётки по Y (дефолт 16)")
    args = parser.parse_args()

    array = ArrayConfig(args.nx, args.ny)

    builder = SceneBuilder()
    synth   = Synthesizer(array, RNG, seed=42)
    model   = Fft3DModel(array, RNG, windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()))
    nuller  = SubspaceNuller(n_jammers=1)
    cfar    = CaCfarDetector(pfa=1e-3, n_train=8, n_guard=4)  # guard перекрывает мейнлоуб

    # ── Синтез сцены ───────────────────────────────────────────────────────────
    scene = builder.build(SCENE_CFG)
    raw   = synth.build(scene)        # (16, 16, 16) complex128

    # ── Нуллинг ────────────────────────────────────────────────────────────────
    cleaned = nuller.apply(raw)       # (16, 16, 16), raw не изменён

    # ── 3D-БПФ ────────────────────────────────────────────────────────────────
    cube_before = model.process(raw)
    cube_after  = model.process(cleaned)

    # ── CFAR ──────────────────────────────────────────────────────────────────
    dets_before = cfar.detect(cube_before)
    dets_after  = cfar.detect(cube_after)

    # ── Заголовок ─────────────────────────────────────────────────────────────
    print("═" * 68)
    print("  CA-CFAR детектор · demo")
    print(f"  Апертура: nx×ny={array.nx}×{array.ny}, padded={array.padded_shape()}")
    print(f"  Сцена: цель kx={TARGET_KX:+d} range={TARGET_RANGE},"
          f" barrage kx={BARRAGE_KX:+d}")
    print(f"  pfa={cfar.pfa:.0e}, n_train={cfar.n_train}, n_guard={cfar.n_guard}")
    print(f"  α = {cfar.alpha:.4f}  (N={2*cfar.n_train} опорных при полном окне)")
    print("═" * 68)

    # ── Отчёт нуллера ─────────────────────────────────────────────────────────
    rep = nuller.report(raw)
    print(f"\n  NullerReport: λ₁/σ_n²={rep.lambda_ratio:.1f},"
          f" occupancy={rep.occupancy:.3f}, is_barrage={rep.is_barrage}")

    # ── Детекции ДО нуллинга ───────────────────────────────────────────────────
    _print_detections(dets_before, "ДО нуллинга", TARGET_KX, TARGET_RANGE, BARRAGE_KX)

    # ── Детекции ПОСЛЕ нуллинга ────────────────────────────────────────────────
    _print_detections(dets_after,  "ПОСЛЕ нуллинга", TARGET_KX, TARGET_RANGE, BARRAGE_KX)

    # ── Итоговый профиль дальности в столбе цели (до/после) ──────────────────
    ix_t, iy_t = cube_before.index_of_angle(TARGET_KX, TARGET_KY)
    ix_b, iy_b = cube_before.index_of_angle(BARRAGE_KX, BARRAGE_KY)

    prof_t_before = cube_before.range_profile_db(ix_t, iy_t)
    prof_t_after  = cube_after.range_profile_db(ix_t, iy_t)

    print(f"\n  Профиль дальности  (в бине кx=цель={TARGET_KX}):")
    print(f"    ДО  нуллинга: max(prof)=0 дБ,"
          f" range[{TARGET_RANGE}]={prof_t_before[TARGET_RANGE]:.1f} дБ")
    print(f"    ПОСЛЕ нуллинга: max(prof)=0 дБ,"
          f" range[{TARGET_RANGE}]={prof_t_after[TARGET_RANGE]:.1f} дБ")

    # Используем raw magnitude для абсолютного сравнения
    rp_b_before = 20.0 * np.log10(cube_before.magnitude[ix_b, iy_b, :] + 1e-12)
    rp_b_after  = 20.0 * np.log10(cube_after.magnitude[ix_b, iy_b,  :] + 1e-12)
    suppression = float(rp_b_before.max() - rp_b_after.max())

    print(f"\n  Профиль дальности  (в бине kx=barrage={BARRAGE_KX}):")
    print(f"    ДО  нуллинга: peak={rp_b_before.max():.1f} дБ")
    print(f"    ПОСЛЕ нуллинга: peak={rp_b_after.max():.1f} дБ")
    print(f"    → подавление помехи: {suppression:.1f} дБ")

    print()
    print("═" * 68)

    # Финальная проверка
    ix_t2, iy_t2 = cube_after.index_of_angle(TARGET_KX, TARGET_KY)
    target_in_after = any(
        d.kx_idx == ix_t2 and d.ky_idx == iy_t2 and abs(d.range_bin - TARGET_RANGE) <= 1
        for d in dets_after
    )
    barrage_fa_after = sum(1 for d in dets_after if d.kx == BARRAGE_KX)

    print(f"  ИТОГ: цель найдена после нуллинга = {target_in_after}"
          f" | ложных в столбе barrage = {barrage_fa_after}")
    print("═" * 68)


if __name__ == "__main__":
    main()
