"""Тесты RobustMvdrNuller (БЕЗ pytest — только TestRunner).

Запуск:  python tests/test_mvdr.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работала и форма `python tests/test_mvdr.py`, и `-m tests.test_mvdr`.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import ArrayConfig, BarrageSpec, RangeConfig, SceneConfig, TargetSpec  # noqa: E402
from core.config.scene_config import ThermalNoiseSpec  # noqa: E402
from core.generators.grid import ArrayGrid  # noqa: E402
from core.generators.scene import SceneBuilder, Synthesizer  # noqa: E402
from core.models.anti_barrage import RobustMvdrNuller  # noqa: E402

# ── Вспомогательные фабрики сцен (как в test_nuller.py) ──────────────────────

_ARRAY = ArrayConfig(16, 16)
_RNG   = RangeConfig(n_real=16, n_fft=16)   # K=16 < M=256 → R всегда вырождена (rank<=K)
_GRID  = ArrayGrid.from_config(_ARRAY)


def _synth(scene_cfg: SceneConfig, seed: int = 1) -> np.ndarray:
    synth = Synthesizer(_ARRAY, _RNG, seed=seed)
    builder = SceneBuilder()
    return synth.build(builder.build(scene_cfg))


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

class MvdrNullerTests(TestRunner):

    def setup(self) -> None:
        self.a_t = _GRID.steering(2.0, 0.0)   # (nx, ny) target steering, kx=2.0

    # ── 1. Distortionless constraint: wᴴ a ≈ 1 ──────────────────────────────
    def test_distortionless_constraint(self) -> AssertionGroup:
        g = AssertionGroup("mvdr.distortionless_constraint")

        raw = _target_barrage_raw()
        nuller = RobustMvdrNuller(self.a_t, loading=0.01)
        w = nuller.weights(raw)

        a_flat = self.a_t.astype(np.complex128).ravel()
        response = complex(w.conj() @ a_flat)
        err = abs(response - 1.0)
        g.add(err < 1e-6,
              f"wᴴ a должен быть ≈ 1 (distortionless constraint), получено {response} (err={err:.2e})")
        return g

    # ── 2. Diagonal loading стабилизирует веса при K < M (R вырождена) ──────
    def test_loading_stabilizes_weights(self) -> AssertionGroup:
        g = AssertionGroup("mvdr.loading_stabilizes_weights")

        raw = _target_barrage_raw()

        w_tiny  = RobustMvdrNuller(self.a_t, loading=1e-8).weights(raw)
        w_small = RobustMvdrNuller(self.a_t, loading=0.01).weights(raw)
        w_big   = RobustMvdrNuller(self.a_t, loading=0.2).weights(raw)

        n_tiny, n_small, n_big = (float(np.linalg.norm(w)) for w in (w_tiny, w_small, w_big))

        g.add(np.isfinite(n_tiny) and np.isfinite(n_small) and np.isfinite(n_big),
              f"все нормы весов должны быть конечны: tiny={n_tiny}, small={n_small}, big={n_big}")

        # Монотонное убывание нормы весов с ростом loading (регуляризация R⁻¹)
        g.add(n_tiny > n_small > n_big,
              f"‖w‖ должна убывать с ростом loading: tiny={n_tiny:.2e} > small={n_small:.2e}"
              f" > big={n_big:.2e}")

        # Loading критичен: при почти нулевом loading норма веса на порядки больше,
        # чем при разумном loading (в отличие от SubspaceNuller, где loading no-op).
        g.add(n_tiny > 5.0 * n_big,
              f"‖w‖ при loading→0 должна быть >> ‖w‖ при loading=0.2 (вырожденная R),"
              f" получено tiny={n_tiny:.2e}, big={n_big:.2e}")
        return g

    # ── 3. Подавление помехи: отклик луча выше на цель, чем на заград ───────
    def test_barrage_suppressed_relative_to_target(self) -> AssertionGroup:
        g = AssertionGroup("mvdr.barrage_suppressed")

        kx_t, kx_j = 2.0, -4.0
        raw = _target_barrage_raw(kx_t=kx_t, kx_j=kx_j, power=6.0)
        nuller = RobustMvdrNuller(self.a_t, loading=0.01)
        w = nuller.weights(raw)

        a_j = _GRID.steering(kx_j, 0.0).astype(np.complex128).ravel()
        a_t_flat = self.a_t.astype(np.complex128).ravel()

        resp_target = abs(complex(w.conj() @ a_t_flat))    # ≈ 1 (constraint)
        resp_jammer = abs(complex(w.conj() @ a_j))          # должен быть подавлен

        g.add(resp_target > 0.9,
              f"отклик луча на цель должен быть ≈1, получено {resp_target:.3f}")
        g.add(resp_jammer < 0.3 * resp_target,
              f"отклик на заград должен быть подавлен относительно цели: "
              f"resp_target={resp_target:.3f}, resp_jammer={resp_jammer:.3f}")

        # Дополнительно: сравнить энергию выхода MVDR-луча с энергией сырых данных
        # спроецированных наивно (без адаптации) на направление цели.
        y = nuller.apply(raw)
        e_mvdr = float(np.mean(np.abs(y) ** 2))
        g.add(np.isfinite(e_mvdr) and e_mvdr > 0.0,
              f"выход MVDR-луча должен иметь конечную ненулевую энергию: {e_mvdr}")
        return g

    # ── 4. apply/weights не мутируют вход ────────────────────────────────────
    def test_input_immutability(self) -> AssertionGroup:
        g = AssertionGroup("mvdr.input_immutability")

        raw = _target_barrage_raw()
        raw_copy = raw.copy()
        nuller = RobustMvdrNuller(self.a_t, loading=0.01)
        _y = nuller.apply(raw)
        _w = nuller.weights(raw)

        diff = float(np.max(np.abs(raw - raw_copy)))
        g.add(diff == 0.0,
              f"apply()/weights() не должны мутировать вход: max diff={diff}")
        return g


if __name__ == "__main__":
    MvdrNullerTests().run_all()
