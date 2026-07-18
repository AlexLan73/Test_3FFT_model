"""Тесты калибровки триажа на синтетическом датасете (гл.4 §4.12, TASK_calibration).

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_calibration.py

Малые размеры (n_per_class~15-20, апертуры 16x16/64x64) -- быстрый прогон,
Монте-Карло-допуски на accuracy/Pfa (не жёсткие пороги, §4.12: "закрывается
экспериментально").
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.models.tokenizer import NOISE, SMEARED, SOURCE  # noqa: E402
from core.models.tokenizer.calibration import TriageCalibrator  # noqa: E402

_APERTURES = [(16, 16), (64, 64)]
_SNR_DB = [10.0, 15.0, 20.0]
_N_PER_CLASS = 18


class CalibrationTests(TestRunner):

    def setup(self) -> None:
        self.cal = TriageCalibrator(apertures=_APERTURES, snr_db_list=_SNR_DB,
                                     n_per_class=_N_PER_CLASS, seed=0)

    def test_build_dataset_non_empty(self) -> AssertionGroup:
        g = AssertionGroup("calibration.build_dataset_non_empty")
        ds = self.cal.build_dataset()
        expected_n = len(_APERTURES) * len(_SNR_DB) * _N_PER_CLASS
        for label in (SOURCE, NOISE, SMEARED):
            g.add(label in ds, f"датасет должен содержать класс {label}")
            g.add(len(ds.get(label, [])) == expected_n,
                  f"{label}: ожидалось {expected_n} сэмплов, получено {len(ds.get(label, []))}")
        return g

    def test_validate_accuracy_and_pfa(self) -> AssertionGroup:
        """Монте-Карло: source/noise accuracy высокая, Pfa шума низкая (§4.12 п.2)."""
        g = AssertionGroup("calibration.validate_accuracy_and_pfa")
        result = self.cal.validate()
        acc = result["accuracy"]
        g.add(acc[SOURCE] > 0.75, f"source-accuracy={acc[SOURCE]:.2f} должна быть > 0.75")
        g.add(acc[NOISE] > 0.75, f"noise-accuracy={acc[NOISE]:.2f} должна быть > 0.75")
        g.add(result["pfa_noise"] < 0.25, f"pfa_noise={result['pfa_noise']:.2f} должна быть < 0.25")
        return g

    def test_class_stats_shape(self) -> AssertionGroup:
        g = AssertionGroup("calibration.class_stats_shape")
        stats = self.cal.class_stats()
        labels = {s.label for s in stats}
        g.add(labels == {SOURCE, NOISE, SMEARED}, f"class_stats должен покрыть 3 класса, получено {labels}")
        for s in stats:
            # (median, p10, p90) -- p10 <= median <= p90 для всех признаков
            for name, triple in (("pr", s.pr), ("hoyer", s.hoyer),
                                  ("main_frac", s.main_frac), ("lobe_ratio", s.lobe_ratio)):
                median, p10, p90 = triple
                g.add(p10 <= median <= p90,
                      f"{s.label}.{name}: p10={p10:.4f} <= median={median:.4f} <= p90={p90:.4f}")
        return g

    def test_invariant_to_aperture_size(self) -> AssertionGroup:
        """source-класс на (16,16) И (64,64) -- метка source в большинстве случаев (F9)."""
        g = AssertionGroup("calibration.invariant_to_aperture_size")
        by_ap = self.cal.validate_by_aperture()
        for ap in _APERTURES:
            g.add(ap in by_ap, f"апертура {ap} должна быть в validate_by_aperture()")
            if ap in by_ap:
                source_acc = by_ap[ap]["accuracy"][SOURCE]
                g.add(source_acc > 0.75,
                      f"апертура {ap}: source-accuracy={source_acc:.2f} должна быть > 0.75 (инвариантность к M)")
        return g

    def test_deterministic_same_seed(self) -> AssertionGroup:
        """Тот же seed -> тот же confusion (детерминизм калибратора)."""
        g = AssertionGroup("calibration.deterministic_same_seed")
        cal_a = TriageCalibrator(apertures=_APERTURES, snr_db_list=_SNR_DB,
                                  n_per_class=_N_PER_CLASS, seed=42)
        cal_b = TriageCalibrator(apertures=_APERTURES, snr_db_list=_SNR_DB,
                                  n_per_class=_N_PER_CLASS, seed=42)
        result_a = cal_a.validate()
        result_b = cal_b.validate()
        g.add(result_a["confusion"] == result_b["confusion"],
              f"confusion должен совпасть при том же seed: {result_a['confusion']} vs {result_b['confusion']}")
        return g

    def test_different_seed_can_differ(self) -> AssertionGroup:
        """Разные seed -- датасеты не идентичны (проверка, что seed реально используется)."""
        g = AssertionGroup("calibration.different_seed_can_differ")
        cal_a = TriageCalibrator(apertures=[(16, 16)], snr_db_list=[15.0], n_per_class=5, seed=1)
        cal_b = TriageCalibrator(apertures=[(16, 16)], snr_db_list=[15.0], n_per_class=5, seed=2)
        ds_a = cal_a.build_dataset()
        ds_b = cal_b.build_dataset()
        g.add(ds_a[SOURCE][0].pr != ds_b[SOURCE][0].pr,
              "датасеты с разным seed не должны совпадать поэлементно")
        return g


if __name__ == "__main__":
    ok = CalibrationTests().run_all()
    sys.exit(0 if ok else 1)
