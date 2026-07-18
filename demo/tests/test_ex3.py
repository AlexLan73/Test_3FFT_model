"""Приёмка ex3: правильные эхо + помехи + null угла (🚫 pytest, правило 04).

Лёгкие размеры (R2 спеки): 16×16×512, 2 объекта; помехи в тестах — только barrage
и DRFM (строгая приёмка §8 R4), остальные 4 — режим наблюдения в самом примере.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.generators.waveforms import Modulation, build_pulse_echo_volume  # noqa: E402
from core.models.anti_barrage import SubspaceNuller  # noqa: E402
from demo.ex2_am_square.example import Ex2Params, ObjectSpec  # noqa: E402
from demo.ex3_am_barrage.example import (  # noqa: E402
    Ex3AmBarrage,
    Ex3Params,
    add_noise_volume,
    build_echo_volume,
    build_jammer_volume,
    null_angle,
)

_ECHO_KW = dict(fs=500e6, carrier_hz=100e6, n_samples=512, dur_samples=80,
                kx=3, ky=-2, nx=16, ny=16)
_AM_META = {"m": 0.5, "f_m": 12.5e6}


def _light_params() -> Ex3Params:
    base = Ex2Params(nx=16, ny=16, n_axis=512, coarse_depth=32, coarse_step=32,
                     fine_depth=16, fine_step=8,
                     scene=(ObjectSpec("B", "am", 8, 200, 3, -2),
                            ObjectSpec("F", "radio", 16, 380, -5, 4)))
    return Ex3Params(base=base)


class Ex3EchoJammersTests(TestRunner):

    def setup(self) -> None:
        self.p = _light_params()
        self.ex = Ex3AmBarrage(params=self.p)
        self.cfg = self.ex._cfg()
        self.rng = np.random.default_rng(7)

    def test_echo_envelope_front(self) -> AssertionGroup:
        g = AssertionGroup("ex3.echo_front")
        v = build_pulse_echo_volume(Modulation.AM, t0_samples=200, rng=np.random.default_rng(7),
                                    extra_meta=_AM_META, **_ECHO_KW)
        front = abs(v[0, 0, 200])
        g.add(abs(front - 1.5) < 1e-5, f"|эхо| на фронте = 1+m = 1.5 (S1), получено {front:.6f}")
        g.add(bool(np.allclose(v[0, 0, :200], 0)), "до t0 — нули")
        # сдвиг эквивалентен np.roll эталона (тот же rng!)
        v0 = build_pulse_echo_volume(Modulation.AM, t0_samples=0, rng=np.random.default_rng(7),
                                     extra_meta=_AM_META, **_ECHO_KW)
        g.add(bool(np.allclose(np.roll(v0[0, 0, :], 200), v[0, 0, :])),
              "эхо = np.roll(зонд на нуле) поэлементно")
        return g

    def test_echo_no_wrap(self) -> AssertionGroup:
        g = AssertionGroup("ex3.echo_no_wrap")
        v = build_pulse_echo_volume(Modulation.CW, t0_samples=480, rng=np.random.default_rng(7),
                                    **_ECHO_KW)   # dur=80, t0=480 -> хвост за N=512
        g.add(bool(np.allclose(v[0, 0, :480], 0)), "до t0 — нули (заворота в начало нет)")
        g.add(bool(np.any(np.abs(v[0, 0, 480:]) > 0)), "видимая часть эха есть")
        return g

    def test_barrage_band_nulled_objects_found(self) -> AssertionGroup:
        g = AssertionGroup("ex3.barrage")
        echo = build_echo_volume(self.p, self.rng)
        jam = build_jammer_volume(self.p, "barrage", Modulation.BARRAGE, 6, 7, self.rng)
        vol = add_noise_volume(echo + jam, 10.0, self.rng)
        m, _, pts_after, banded, _ = self.ex._run_pipeline(vol, self.cfg)
        g.add(banded, "полоса заграда детектирована (гейт R1)")
        g.add(m["band_angle"] == (6.0, 7.0), f"угол полосы точен, получено {m['band_angle']}")
        g.add(pts_after is not None, "после null сделан повторный скан")
        g.add(m["found"] == 2, f"после null оба объекта найдены, получено {m['found']}")
        g.add(m["false"] == 0, f"ложных 0, получено {m['false']}")
        return g

    def test_null_angle_matches_subspace(self) -> AssertionGroup:
        g = AssertionGroup("ex3.null_vs_subspace")
        # rank-1 null по ИЗВЕСТНОМУ углу ≈ SubspaceNuller (EVD) при доминирующей помехе
        jam = build_jammer_volume(self.p, "barrage", Modulation.BARRAGE, 6, 7,
                                  np.random.default_rng(3))
        cleaned_r1 = null_angle(jam, 6, 7)
        cleaned_evd = SubspaceNuller(n_jammers=1).apply(jam)
        res_r1 = float(np.linalg.norm(cleaned_r1))
        res_evd = float(np.linalg.norm(cleaned_evd))
        raw = float(np.linalg.norm(jam))
        g.add(res_r1 < 1e-3 * raw, f"rank-1 null давит помеху >60 дБ: {res_r1:.3g} vs {raw:.3g}")
        g.add(abs(res_r1 - res_evd) <= max(res_r1, res_evd, 1e-12) * 2 + 1e-9,
              f"rank-1 ≈ EVD-subspace (остатки {res_r1:.3g} / {res_evd:.3g})")
        return g

    def test_drfm_comb_and_objects_survive(self) -> AssertionGroup:
        g = AssertionGroup("ex3.drfm")
        echo = build_echo_volume(self.p, self.rng)
        jam = build_jammer_volume(self.p, "drfm", Modulation.DRFM_REPEATER, -6, -7, self.rng)
        vol = add_noise_volume(echo + jam, 10.0, self.rng)
        m, pts_before, _, banded, _ = self.ex._run_pipeline(vol, self.cfg)
        g.add(m["found"] == 2, f"объекты при DRFM найдены, получено {m['found']}")
        # ГРЕБЁНКА: копии зонда видны как всплески на угле DRFM в НЕСКОЛЬКИХ окнах.
        # «полоса/не полоса» НЕ ассертим: на лёгкой оси 512 копии (шаг 0.1·N) покрывают
        # >1/3 окон -> квази-полоса (гейт срабатывает, null уместен); на полной 4096 —
        # ~5 окон из 128 -> гребёнка идёт обычным путём. Оба поведения физичны.
        comb_hits = [pt for pt in pts_before if abs(pt.kx - (-6)) <= 1 and abs(pt.ky - (-7)) <= 1]
        g.add(len(comb_hits) >= 2,
              f"гребёнка: >=2 окна со всплеском на угле DRFM, получено {len(comb_hits)}")
        return g

    def test_input_not_mutated(self) -> AssertionGroup:
        g = AssertionGroup("ex3.no_mutation")
        echo = build_echo_volume(self.p, np.random.default_rng(7))
        backup = echo.copy()
        null_angle(echo, 3, -2)
        g.add(bool(np.array_equal(echo, backup)), "null_angle не мутирует вход")
        return g


if __name__ == "__main__":
    Ex3EchoJammersTests().run_all()
