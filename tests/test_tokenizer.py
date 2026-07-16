"""Тесты объёмного токенизатора (гл.4 §4.5-4.9, гл.4-бис §4-бис.3, TASK_tokenizer_p1).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_tokenizer.py

Синтетические угловые карты 16x16 (Хэмминг-апертура + комплексный гаусс. шум,
фикс. seed) -- реюз `core.models.angular_fft.angular_fft` (тот же угловой FFT,
что в реальном тракте), НЕ отдельная копия FFT-плюмбинга.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.models.angular_fft import angular_fft  # noqa: E402
from core.models.result import Axis, SpectralCube  # noqa: E402
from core.models.tokenizer import (  # noqa: E402
    BARRAGE,
    COMB,
    NOISE,
    SMEARED,
    SOURCE,
    TARGET,
    FeatureExtractor,
    FeatureVector,
    PeakInfo,
    RuleBasedTriage,
    SliceToken,
    VolumeTokenizer,
    assemble_range,
)
from core.models.windows import HammingWindow  # noqa: E402

_NX = _NY = 16


# ── Синтетические угловые карты (Хэмминг-апертура + компл. гаусс. шум) ────────

def _plane_wave(kx0: float, ky0: float) -> np.ndarray:
    x = np.arange(_NX)[:, None]
    y = np.arange(_NY)[None, :]
    return np.exp(2j * np.pi * (kx0 * x / _NX + ky0 * y / _NY))


def _noise_field(sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((_NX, _NY)) +
            1j * rng.standard_normal((_NX, _NY))) * sigma / np.sqrt(2.0)


def _angular_power(aperture: np.ndarray) -> np.ndarray:
    """P = |A|² после углового FFT (Хэмминг по апертуре) -- реюз `angular_fft`."""
    cube3 = aperture[:, :, None].astype(np.complex128)
    spectrum = angular_fft(cube3, aperture_window=HammingWindow())
    a = spectrum[:, :, 0]
    return a.real ** 2 + a.imag ** 2


def _noise_scene(seed: int = 0) -> np.ndarray:
    return _angular_power(_noise_field(1.0, seed))


def _target_scene(kx0: float = 2.0, ky0: float = -3.0, amp: float = 6.0,
                   seed: int = 10) -> np.ndarray:
    s = amp * _plane_wave(kx0, ky0) + _noise_field(1.0, seed)
    return _angular_power(s)


def _barrage_scene(seed: int = 20, k: int = 14, spread: float = 3.5,
                    base_kx: float = 2.0, base_ky: float = -3.0) -> np.ndarray:
    """"Заградка": K близких когерентных источников со случайной фазой (смазанный лепесток)."""
    s = _noise_field(1.0, seed).copy()
    rng = np.random.default_rng(300 + k * 10 + int(spread * 10))
    for _ in range(k):
        dkx = rng.uniform(-spread, spread)
        dky = rng.uniform(-spread, spread)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        a = 5.0 * rng.uniform(0.7, 1.3)
        s = s + a * np.exp(1j * phase) * _plane_wave(base_kx + dkx, base_ky + dky)
    return _angular_power(s)


# ── 1+2. Разделение классов + страддл-устойчивость (гл.4 §4.11) ──────────────

class FeatureSeparationTests(TestRunner):

    def setup(self) -> None:
        self.extractor = FeatureExtractor()
        self.triage = RuleBasedTriage()

    def test_three_classes_in_corridor(self) -> AssertionGroup:
        """PR/Hoyer/MainFrac/LobeRatio/MaxMean должны попасть в коридоры §4.11."""
        g = AssertionGroup("features.three_classes_in_corridor")

        f_noise = self.extractor.extract(_noise_scene(seed=1))
        f_target = self.extractor.extract(_target_scene(seed=11))
        f_barrage = self.extractor.extract(_barrage_scene(seed=21))

        # шум -- около M/2=128, размазан, низкий контраст
        g.add(f_noise.pr > 60.0, f"шум: PR={f_noise.pr:.1f} должен быть > 60 (табл. ~129)")
        g.add(f_noise.hoyer < 0.55, f"шум: Hoyer={f_noise.hoyer:.2f} должен быть < 0.55 (табл. ~0.31)")
        g.add(f_noise.main_frac < 0.25, f"шум: MainFrac={f_noise.main_frac:.2f} должен быть < 0.25 (табл. ~0.07)")
        g.add(f_noise.max_mean < 15.0, f"шум: MaxMean={f_noise.max_mean:.1f} должен быть < 15 (табл. ~5.4)")

        # цель -- компактный лепесток, высокая собранность
        g.add(f_target.pr < 8.0, f"цель: PR={f_target.pr:.2f} должен быть < 8 (табл. ~3.6)")
        g.add(f_target.hoyer > 0.85, f"цель: Hoyer={f_target.hoyer:.2f} должен быть > 0.85 (табл. ~0.94)")
        g.add(f_target.main_frac > 0.85, f"цель: MainFrac={f_target.main_frac:.2f} должен быть > 0.85 (табл. ~0.98)")
        g.add(f_target.lobe_ratio < 0.05, f"цель: LobeRatio={f_target.lobe_ratio:.4f} должен быть < 0.05 (табл. ~0.002)")
        g.add(f_target.max_mean > 60.0, f"цель: MaxMean={f_target.max_mean:.1f} должен быть > 60 (табл. ~123)")

        # заградка -- промежуточная между целью и шумом
        g.add(8.0 < f_barrage.pr < 40.0, f"заградка: PR={f_barrage.pr:.2f} должен быть в (8,40) (табл. 15-23)")
        g.add(0.6 < f_barrage.hoyer < 0.9, f"заградка: Hoyer={f_barrage.hoyer:.2f} должен быть в (0.6,0.9) (табл. ~0.81)")
        g.add(0.2 < f_barrage.main_frac < 0.65, f"заградка: MainFrac={f_barrage.main_frac:.2f} должен быть в (0.2,0.65) (табл. ~0.40)")
        g.add(15.0 < f_barrage.max_mean < 55.0, f"заградка: MaxMean={f_barrage.max_mean:.1f} должен быть в (15,55) (табл. 22-42)")

        # структурный порядок (важнее абсолютных значений -- патент §4.11: "иллюстративны")
        g.add(f_target.pr < f_barrage.pr < f_noise.pr,
              f"порядок PR: target({f_target.pr:.1f}) < barrage({f_barrage.pr:.1f}) < noise({f_noise.pr:.1f})")
        g.add(f_noise.hoyer < f_barrage.hoyer < f_target.hoyer,
              f"порядок Hoyer: noise({f_noise.hoyer:.2f}) < barrage({f_barrage.hoyer:.2f}) < target({f_target.hoyer:.2f})")
        g.add(f_noise.main_frac < f_barrage.main_frac < f_target.main_frac,
              "порядок MainFrac: noise < barrage < target")
        return g

    def test_rule_based_triage_labels(self) -> AssertionGroup:
        """RuleBasedTriage должен разметить {noise/source/smeared} верно."""
        g = AssertionGroup("features.rule_based_triage_labels")
        cases = [
            ("noise", _noise_scene(seed=2), NOISE),
            ("target", _target_scene(seed=12), SOURCE),
            ("barrage", _barrage_scene(seed=22), SMEARED),
        ]
        for name, scene, expected in cases:
            f = self.extractor.extract(scene)
            label, score = self.triage.classify(f)
            g.add(label == expected, f"{name}: label={label} должен быть {expected} (score={score:.2f})")
        return g

    def test_straddle_robustness(self) -> AssertionGroup:
        """Цель на полбина -- PR/Hoyer/MainFrac/LobeRatio почти не плывут (§4.11 табл.2)."""
        g = AssertionGroup("features.straddle_robustness")
        f_grid = self.extractor.extract(_target_scene(kx0=2.0, ky0=-3.0, seed=13))
        f_straddle = self.extractor.extract(_target_scene(kx0=2.5, ky0=-3.5, seed=13))

        def rel(a: float, b: float) -> float:
            return abs(a - b) / max(abs(a), 1e-9)

        g.add(rel(f_grid.pr, f_straddle.pr) < 0.6,
              f"PR устойчив: сетка={f_grid.pr:.2f} полбина={f_straddle.pr:.2f}")
        g.add(rel(f_grid.hoyer, f_straddle.hoyer) < 0.15,
              f"Hoyer устойчив: сетка={f_grid.hoyer:.3f} полбина={f_straddle.hoyer:.3f}")
        g.add(rel(f_grid.main_frac, f_straddle.main_frac) < 0.15,
              f"MainFrac устойчив: сетка={f_grid.main_frac:.3f} полбина={f_straddle.main_frac:.3f}")
        g.add(f_straddle.lobe_ratio < 0.05,
              f"LobeRatio остаётся малым при страддле: {f_straddle.lobe_ratio:.4f} (табл. ~0.002)")

        # RAW "2-й/1-й пик" (сырой) -- НЕ используем, но проверим что он ломается (§4.11)
        p_grid = _target_scene(kx0=2.0, ky0=-3.0, seed=13)
        p_straddle = _target_scene(kx0=2.5, ky0=-3.5, seed=13)
        raw_grid = _raw_second_over_first(p_grid)
        raw_straddle = _raw_second_over_first(p_straddle)
        g.add(abs(raw_grid - raw_straddle) > 0.3,
              f"сырой 2й/1й пик ДОЛЖЕН плыть (демонстрация, почему не используется): "
              f"сетка={raw_grid:.3f} полбина={raw_straddle:.3f}")
        return g


def _raw_second_over_first(p: np.ndarray) -> float:
    """"Сырой" 2-й/1-й пик -- top-2 ЕДИНИЧНЫХ ячейки (НЕ используется токенизатором)."""
    flat = np.sort(p.ravel())[::-1]
    return float(flat[1] / max(flat[0], 1e-30))


# ── 3. Проход 2 -- сборка по дальности (target/comb/barrage, гл.4 §4.9) ──────

class RangeAssemblyTests(TestRunner):

    _F = FeatureVector(pr=1.0, hoyer=1.0, main_frac=1.0, lobe_ratio=0.0, max_mean=1.0, energy=1.0)

    def _tok(self, r: int, kx: float = 2.0, ky: float = -3.0) -> SliceToken:
        peak = PeakInfo(kx=kx, ky=ky, amp=1.0, edge=0.0)
        return SliceToken(r=r, peaks=(peak,), f=self._F, label=SOURCE, score=0.9)

    def test_single_token_is_target(self) -> AssertionGroup:
        g = AssertionGroup("range.single_token_is_target")
        verdicts = assemble_range([self._tok(5)])
        g.add(len(verdicts) == 1, f"должен быть 1 verdict, получено {len(verdicts)}")
        if verdicts:
            v = verdicts[0]
            g.add(v.kind == TARGET, f"одиночный токен -> target, получено {v.kind}")
            g.add(v.lead_r == 5, f"lead_r должен быть 5, получено {v.lead_r}")
            g.add(v.period_dr is None, "period_dr должен быть None для одиночной цели")
        return g

    def test_regular_chain_is_comb(self) -> AssertionGroup:
        g = AssertionGroup("range.regular_chain_is_comb")
        tokens = [self._tok(r) for r in (4, 8, 12, 16)]
        verdicts = assemble_range(tokens)
        g.add(len(verdicts) == 1, f"должен быть 1 verdict, получено {len(verdicts)}")
        if verdicts:
            v = verdicts[0]
            g.add(v.kind == COMB, f"регулярная цепочка Δr=4 -> comb, получено {v.kind}")
            g.add(v.lead_r == 4, f"lead_r должен быть min(r)=4, получено {v.lead_r}")
            g.add(v.period_dr == 4.0, f"period_dr должен быть 4.0, получено {v.period_dr}")
        return g

    def test_dense_run_is_barrage(self) -> AssertionGroup:
        g = AssertionGroup("range.dense_run_is_barrage")
        tokens = [self._tok(r) for r in range(10)]   # 0..9 подряд -- сплошной источник
        verdicts = assemble_range(tokens)
        g.add(len(verdicts) == 1, f"должен быть 1 verdict, получено {len(verdicts)}")
        if verdicts:
            v = verdicts[0]
            g.add(v.kind == BARRAGE, f"токены во всех r подряд -> barrage, получено {v.kind}")
            g.add(v.lead_r == 0, f"lead_r должен быть 0, получено {v.lead_r}")
        return g

    def test_non_source_tokens_ignored(self) -> AssertionGroup:
        g = AssertionGroup("range.non_source_tokens_ignored")
        smeared_tok = SliceToken(r=3, peaks=(PeakInfo(kx=0.0, ky=0.0, amp=1.0, edge=0.0),),
                                  f=self._F, label=SMEARED, score=0.5)
        verdicts = assemble_range([smeared_tok])
        g.add(len(verdicts) == 0, f"smeared-токены не участвуют в сборке, получено {len(verdicts)} verdicts")
        return g

    def test_different_angles_are_separate_groups(self) -> AssertionGroup:
        g = AssertionGroup("range.different_angles_are_separate_groups")
        tokens = [self._tok(5, kx=2.0, ky=-3.0), self._tok(7, kx=-4.0, ky=1.0)]
        verdicts = assemble_range(tokens)
        g.add(len(verdicts) == 2, f"разные углы -> разные группы, получено {len(verdicts)}")
        g.add(all(v.kind == TARGET for v in verdicts), "обе группы -- одиночные target")
        return g


# ── 4. L=1 (VolumeTokenizer) эквивалентен плоскому пути ──────────────────────

class VolumeTokenizerTests(TestRunner):

    def _cube(self, n_range: int = 6, target_r: int = 3, seed_base: int = 100) -> SpectralCube:
        mags = np.empty((_NX, _NY, n_range), dtype=np.float64)
        for r in range(n_range):
            p = _target_scene(seed=seed_base + r) if r == target_r else _noise_scene(seed=seed_base + r)
            mags[:, :, r] = np.sqrt(p)     # cube.magnitude хранит |A|, не |A|² (E7)
        kx = Axis("kx", np.arange(-_NX // 2, _NX // 2), centered=True)
        ky = Axis("ky", np.arange(-_NY // 2, _NY // 2), centered=True)
        rng = Axis("range", np.arange(n_range) * 5.0, centered=False)
        return SpectralCube(mags, kx, ky, rng)

    def test_l1_matches_plain_path(self) -> AssertionGroup:
        g = AssertionGroup("tokenizer.l1_matches_plain_path")
        cube = self._cube()
        tokenizer = VolumeTokenizer(window_l=1)
        tokens = tokenizer.tokenize(cube)

        extractor = FeatureExtractor()
        triage = RuleBasedTriage()
        n_range = cube.magnitude.shape[2]
        expected_written = []
        for r in range(n_range):
            power = cube.magnitude[:, :, r].astype(np.float64) ** 2
            f = extractor.extract(power)
            label, _score = triage.classify(f)
            if label != NOISE:
                expected_written.append((r, label, f))

        g.add(len(tokens) == len(expected_written),
              f"число НЕ-шумовых токенов должно совпасть: {len(tokens)} vs {len(expected_written)}")
        by_r = {t.r: t for t in tokens}
        for r, label, f in expected_written:
            g.add(r in by_r, f"токен на r={r} должен присутствовать")
            if r in by_r:
                tok = by_r[r]
                g.add(tok.label == label, f"r={r}: label {tok.label} должен быть {label}")
                g.add(abs(tok.f.pr - f.pr) < 1e-9, f"r={r}: PR токена должен совпасть с плоским путём")
                g.add(abs(tok.f.main_frac - f.main_frac) < 1e-9,
                      f"r={r}: MainFrac токена должен совпасть с плоским путём")
        return g

    def test_target_slice_has_single_peak(self) -> AssertionGroup:
        g = AssertionGroup("tokenizer.target_slice_has_single_peak")
        cube = self._cube(target_r=3)
        tokens = VolumeTokenizer(window_l=1).tokenize(cube)
        by_r = {t.r: t for t in tokens}
        g.add(3 in by_r, "токен на r=3 (цель) должен присутствовать")
        if 3 in by_r:
            tok = by_r[3]
            g.add(tok.label == SOURCE, f"r=3 должен быть source, получено {tok.label}")
            g.add(tok.n_peaks >= 1, f"должен быть хотя бы 1 пик, получено {tok.n_peaks}")
        return g

    def test_sparse_output_skips_noise(self) -> AssertionGroup:
        """Пустые/шумовые срезы НЕ пишутся -- разрежённый выход (§4.7)."""
        g = AssertionGroup("tokenizer.sparse_output_skips_noise")
        cube = self._cube(n_range=8, target_r=4)
        tokens = VolumeTokenizer(window_l=1).tokenize(cube)
        g.add(len(tokens) < cube.magnitude.shape[2],
              f"токенов ({len(tokens)}) должно быть меньше, чем срезов ({cube.magnitude.shape[2]}) -- "
              f"шумовые срезы не пишутся")
        g.add(all(t.label != NOISE for t in tokens), "среди записанных токенов не должно быть noise")
        return g

    def test_does_not_mutate_cube(self) -> AssertionGroup:
        g = AssertionGroup("tokenizer.does_not_mutate_cube")
        cube = self._cube()
        before = cube.magnitude.copy()
        _ = VolumeTokenizer(window_l=1).tokenize(cube)
        diff = float(np.max(np.abs(cube.magnitude - before)))
        g.add(diff == 0.0, f"tokenize() не должен мутировать куб: max diff={diff}")
        return g

    def test_volume_window_l_greater_1_runs(self) -> AssertionGroup:
        """Объёмный путь (L>1, гл.4-бис): не падает, отдаёт корректно сформированные токены."""
        g = AssertionGroup("tokenizer.volume_window_l_greater_1_runs")
        cube = self._cube(n_range=8, target_r=3)
        tokenizer = VolumeTokenizer(window_l=3)
        tokens = tokenizer.tokenize(cube)
        g.add(len(tokens) >= 0, "токенизация L=3 не должна падать")
        for t in tokens:
            g.add(0 <= t.r <= cube.magnitude.shape[2] - 3, f"r={t.r} должен быть валидным началом окна L=3")
            g.add(t.label in (SOURCE, SMEARED), f"непустой токен должен иметь label != noise, получено {t.label}")
        return g


if __name__ == "__main__":
    ok = True
    for cls in (FeatureSeparationTests, RangeAssemblyTests, VolumeTokenizerTests):
        ok = cls().run_all() and ok
    sys.exit(0 if ok else 1)
