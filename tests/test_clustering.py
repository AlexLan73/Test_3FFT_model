"""Тесты угловой кластеризации CFAR-детекций (БЕЗ pytest -- только TestRunner).

Запуск:  python tests/test_clustering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.models.anti_barrage import Detection, DetectionClusterer  # noqa: E402


def _det(kx_idx: int, ky_idx: int, range_bin: int, level_db: float,
         kx: float, ky: float) -> Detection:
    """Короткий конструктор Detection для тестов (threshold_db -- произвольный)."""
    return Detection(
        kx_idx=kx_idx, ky_idx=ky_idx, range_bin=range_bin,
        level_db=level_db, threshold_db=level_db - 3.0,
        kx=kx, ky=ky,
    )


class ClusteringTests(TestRunner):

    def setup(self) -> None:
        self.clusterer = DetectionClusterer(angle_tol=1.0, range_tol=1)

    # ── 1. Три соседние детекции одного источника -> один кластер ─────────────
    def test_neighbors_merge_into_one_cluster(self) -> AssertionGroup:
        g = AssertionGroup("clustering.neighbors_merge")
        dets = [
            _det(2, 0, 8,  -6.0, kx=2.0, ky=0.0),
            _det(2, 0, 9,  -1.0, kx=2.0, ky=0.0),   # пик
            _det(2, 0, 10, -5.5, kx=2.0, ky=0.0),
        ]
        clusters = self.clusterer.cluster(dets)
        g.add(len(clusters) == 1, f"ожидается 1 кластер, получено {len(clusters)}")
        if clusters:
            c = clusters[0]
            g.add(c.n_members == 3, f"n_members должно = 3, получено {c.n_members}")
            g.add(c.range_bin == 9, f"центроид (пик) range_bin должен = 9, получено {c.range_bin}")
            g.add(abs(c.peak_level_db - (-1.0)) < 1e-9,
                  f"peak_level_db должен = -1.0, получено {c.peak_level_db}")
        return g

    # ── 2. Две далёкие группы -> два кластера ──────────────────────────────────
    def test_two_far_groups_stay_separate(self) -> AssertionGroup:
        g = AssertionGroup("clustering.two_far_groups")
        dets = [
            _det(2, 0, 8,  -3.0, kx=2.0, ky=0.0),
            _det(2, 0, 9,  -2.0, kx=2.0, ky=0.0),   # пик группы A
            _det(9, 5, 30, -4.0, kx=10.0, ky=5.0),  # пик группы B
            _det(9, 5, 31, -5.0, kx=10.0, ky=5.0),
        ]
        clusters = self.clusterer.cluster(dets)
        g.add(len(clusters) == 2, f"ожидается 2 кластера, получено {len(clusters)}")
        n_members_sorted = sorted(c.n_members for c in clusters)
        g.add(n_members_sorted == [2, 2],
              f"обе группы по 2 детекции, получено {n_members_sorted}")
        return g

    # ── 3. Одиночная детекция -> кластер n_members=1 ───────────────────────────
    def test_single_detection_is_own_cluster(self) -> AssertionGroup:
        g = AssertionGroup("clustering.single_detection")
        dets = [_det(0, 0, 5, -3.0, kx=0.0, ky=0.0)]
        clusters = self.clusterer.cluster(dets)
        g.add(len(clusters) == 1, f"ожидается 1 кластер, получено {len(clusters)}")
        if clusters:
            g.add(clusters[0].n_members == 1,
                  f"n_members должно = 1, получено {clusters[0].n_members}")
        return g

    # ── 4. Пустой список -> пустой результат ───────────────────────────────────
    def test_empty_input_gives_empty_output(self) -> AssertionGroup:
        g = AssertionGroup("clustering.empty_input")
        clusters = self.clusterer.cluster([])
        g.add(clusters == [], f"ожидается пустой список, получено {clusters}")
        return g

    # ── 5. cluster() не мутирует вход ──────────────────────────────────────────
    def test_input_not_mutated(self) -> AssertionGroup:
        g = AssertionGroup("clustering.input_not_mutated")
        dets = [
            _det(2, 0, 9, -1.0, kx=2.0, ky=0.0),
            _det(0, 0, 1, -8.0, kx=0.0, ky=0.0),
        ]
        before = list(dets)
        _ = self.clusterer.cluster(dets)
        g.add(dets == before, "исходный список детекций не должен измениться")
        g.add(dets is not before or dets == before, "sanity: сравнение содержимого")
        return g

    # ── 6. Smoke: 5 детекций (3 соседние + 2 отдельные) -> 3 кластера ──────────
    def test_smoke_five_detections_three_clusters(self) -> AssertionGroup:
        g = AssertionGroup("clustering.smoke_5_to_3")
        dets = [
            _det(2, 0, 8,  -6.0, kx=2.0, ky=0.0),
            _det(2, 0, 9,  -1.0, kx=2.0, ky=0.0),   # пик группы соседних
            _det(2, 0, 10, -5.5, kx=2.0, ky=0.0),
            _det(9, 5, 30, -4.0, kx=10.0, ky=5.0),  # одиночка B
            _det(-3, -3, 50, -7.0, kx=-6.0, ky=-6.0),  # одиночка C
        ]
        clusters = self.clusterer.cluster(dets)
        g.add(len(clusters) == 3, f"ожидается 3 кластера, получено {len(clusters)}")
        n_members_sorted = sorted(c.n_members for c in clusters)
        g.add(n_members_sorted == [1, 1, 3],
              f"ожидается [1,1,3] членов, получено {n_members_sorted}")
        return g


if __name__ == "__main__":
    ClusteringTests().run_all()
