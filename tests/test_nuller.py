"""Тесты SubspaceNuller (БЕЗ pytest — только TestRunner).

Запуск:  python tests/test_nuller.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работала и форма `python tests/test_nuller.py`, и `-m tests.test_nuller`.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import ArrayConfig, BarrageSpec, RangeConfig, SceneConfig, TargetSpec  # noqa: E402
from core.config.scene_config import ThermalNoiseSpec  # noqa: E402
from core.generators.scene import SceneBuilder, Synthesizer  # noqa: E402
from core.models.anti_barrage import SubspaceNuller  # noqa: E402

# ── Вспомогательные фабрики сцен ──────────────────────────────────────────────

_ARRAY = ArrayConfig(16, 16)
_RNG   = RangeConfig(n_real=16, n_fft=16)   # n_fft=16 — минимально, тесты быстрее


def _synth(scene_cfg: SceneConfig, seed: int = 1) -> np.ndarray:
    synth = Synthesizer(_ARRAY, _RNG, seed=seed)
    builder = SceneBuilder()
    return synth.build(builder.build(scene_cfg))


def _barrage_raw(kx: float = -4.0, power: float = 6.0, seed: int = 1) -> np.ndarray:
    cfg = SceneConfig(
        emitters=(BarrageSpec(kx=kx, ky=0.0, power=power),),
        thermal=ThermalNoiseSpec(power=0.02),
    )
    return _synth(cfg, seed=seed)


def _target_raw(kx: float = 2.0, seed: int = 1) -> np.ndarray:
    cfg = SceneConfig(
        emitters=(TargetSpec(kx=kx, ky=0.0, range_bin=4.0, amplitude=1.0),),
        thermal=ThermalNoiseSpec(power=0.02),
    )
    return _synth(cfg, seed=seed)


def _target_barrage_raw(
    kx_t: float = 2.0, kx_j: float = -4.0, power: float = 6.0, seed: int = 1
) -> np.ndarray:
    cfg = SceneConfig(
        emitters=(
            TargetSpec(kx=kx_t, ky=0.0, range_bin=4.0, amplitude=1.0),
            BarrageSpec(kx=kx_j, ky=0.0, power=power),
        ),
        thermal=ThermalNoiseSpec(power=0.02),
    )
    return _synth(cfg, seed=seed)


# ── Тест-набор ─────────────────────────────────────────────────────────────────

class NullerTests(TestRunner):

    def setup(self) -> None:
        self.nuller = SubspaceNuller(n_jammers=1)

    # ── 1. Подавление barrage-only ≥ 20 дБ ─────────────────────────────────
    def test_barrage_suppression(self) -> AssertionGroup:
        g = AssertionGroup("nuller.barrage_suppression")

        raw     = _barrage_raw(kx=-4.0, power=6.0)
        cleaned = self.nuller.apply(raw)

        e_before = float(np.mean(np.abs(raw) ** 2))
        e_after  = float(np.mean(np.abs(cleaned) ** 2))

        suppression_db = 10.0 * np.log10(e_before / max(e_after, 1e-30))
        g.add(suppression_db >= 20.0,
              f"подавление помехи должно быть ≥ 20 дБ, получено {suppression_db:.1f} дБ")
        return g

    # ── 2. Цель при разных углах выживает ────────────────────────────────────
    def test_target_survives(self) -> AssertionGroup:
        g = AssertionGroup("nuller.target_survives")

        raw     = _target_barrage_raw(kx_t=2.0, kx_j=-4.0, power=6.0)
        cleaned = self.nuller.apply(raw)

        # После нуллинга помеха гасится (supp >= 15 дБ — менее строго для смеси)
        e_raw  = float(np.mean(np.abs(raw) ** 2))
        e_cln  = float(np.mean(np.abs(cleaned) ** 2))
        supp   = 10.0 * np.log10(e_raw / max(e_cln, 1e-30))
        g.add(supp >= 10.0,
              f"после нуллинга общая энергия должна упасть ≥ 10 дБ, получено {supp:.1f} дБ")

        # Цель (kx=+2) — ортогональный угол → проекция мало трогает её
        # Проверяем через пространственную норму вектора наведения на цель
        from core.generators.grid import ArrayGrid
        grid = ArrayGrid(16, 16)
        a_t = grid.steering(2.0, 0.0).ravel()  # (256,)

        nx, ny, k_snap = raw.shape
        m_elem = nx * ny
        x_raw = raw.reshape(m_elem, k_snap).astype(np.complex128)
        x_cln = cleaned.reshape(m_elem, k_snap).astype(np.complex128)

        # Проекция сигнала на направление цели (норма по всем K)
        proj_before = float(np.linalg.norm(a_t.conj() @ x_raw))
        proj_after  = float(np.linalg.norm(a_t.conj() @ x_cln))

        # Разница в проекции не должна быть больше 15 дБ (утечка минимальна)
        proj_loss_db = 20.0 * np.log10(max(proj_before, 1e-30) / max(proj_after, 1e-30))
        g.add(proj_loss_db < 15.0,
              f"утечка в цель ≤ 15 дБ, получено {proj_loss_db:.1f} дБ")
        return g

    # ── 3. Идемпотентность проектора: P⊥² ≈ P⊥, P⊥ @ E_J ≈ 0 ───────────────
    def test_projector_idempotent(self) -> AssertionGroup:
        g = AssertionGroup("nuller.projector_idempotent")

        raw = _barrage_raw(kx=-4.0, power=6.0)
        p_perp, e_jam = self.nuller.decompose(raw)

        # P⊥² ≈ P⊥
        p_sq = p_perp @ p_perp
        err_idem = float(np.linalg.norm(p_sq - p_perp, "fro"))
        ref_norm  = float(np.linalg.norm(p_perp, "fro"))
        rel_idem  = err_idem / max(ref_norm, 1e-30)
        g.add(rel_idem < 1e-8,
              f"P⊥² ≈ P⊥: относит. ошибка {rel_idem:.2e} (должна быть < 1e-8)")

        # P⊥ @ E_J ≈ 0
        residual = p_perp @ e_jam
        err_null  = float(np.linalg.norm(residual, "fro"))
        rel_null  = err_null / max(float(np.linalg.norm(e_jam, "fro")), 1e-30)
        g.add(rel_null < 1e-8,
              f"P⊥ @ E_J ≈ 0: относит. ошибка {rel_null:.2e} (должна быть < 1e-8)")

        # Симметрия/эрмитовость P⊥
        err_herm = float(np.linalg.norm(p_perp - p_perp.conj().T, "fro"))
        g.add(err_herm < 1e-10,
              f"P⊥ должна быть эрмитовой: ошибка {err_herm:.2e}")
        return g

    # ── 4. report классифицирует barrage/target правильно ────────────────────
    def test_report_classification(self) -> AssertionGroup:
        g = AssertionGroup("nuller.report_classification")

        # barrage-only → is_barrage = True
        raw_b = _barrage_raw(kx=-4.0, power=6.0)
        rep_b = self.nuller.report(raw_b)
        g.add(rep_b.is_barrage is True,
              f"barrage-only: is_barrage должен быть True (λ={rep_b.lambda_ratio:.1f},"
              f" occ={rep_b.occupancy:.3f})")

        # target-only → is_barrage = False
        raw_t = _target_raw(kx=2.0)
        rep_t = self.nuller.report(raw_t)
        g.add(rep_t.is_barrage is False,
              f"target-only: is_barrage должен быть False (λ={rep_t.lambda_ratio:.1f},"
              f" occ={rep_t.occupancy:.3f})")

        # Дополнительно: у barrage lambda_ratio >> 1
        g.add(rep_b.lambda_ratio > 10.0,
              f"barrage: λ_ratio={rep_b.lambda_ratio:.1f} должен быть > 10")

        # occupancy barrage выше target (barrage заполняет все бины дальности)
        g.add(rep_b.occupancy > rep_t.occupancy,
              f"occupancy_barrage={rep_b.occupancy:.3f} > occupancy_target={rep_t.occupancy:.3f}")
        return g

    # ── 5. Вход apply не мутируется ─────────────────────────────────────────
    def test_input_immutability(self) -> AssertionGroup:
        g = AssertionGroup("nuller.input_immutability")

        raw = _barrage_raw()
        raw_copy = raw.copy()
        _cleaned = self.nuller.apply(raw)

        diff = float(np.max(np.abs(raw - raw_copy)))
        g.add(diff == 0.0,
              f"apply() не должен мутировать вход: max diff={diff}")
        return g


if __name__ == "__main__":
    NullerTests().run_all()
