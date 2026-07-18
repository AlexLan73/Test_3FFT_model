"""Угловая кластеризация сырых CFAR-детекций (§7 Doc/anti_barrage_math.md, phase2).

Один физический источник (цель/заград) из-за ширины лепестка/страддла по
дальности и углу засвечивает НЕСКОЛЬКО соседних ячеек (kx, ky, range_bin) →
CA-CFAR (`CaCfarDetector`) выдаёт несколько `Detection` на один источник.
`DetectionClusterer` группирует близкие детекции в один `DetectionCluster`
(центроид = пик по level_db) — убирает дубли, даёт по одному кандидату на
источник (чище вход для трекинга).

Алгоритм: single-linkage по связности (граф близости → связные компоненты).
Две детекции соседние, если одновременно:
    |Δkx| <= angle_tol  И  |Δky| <= angle_tol  И  |Δrange_bin| <= range_tol
Кластер = транзитивное замыкание отношения соседства (union-find).
Детерминировано: детекции и результирующие кластеры сортируются по
(range_bin, kx, ky) перед обработкой/выдачей.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cfar import Detection

# ── Value Object — кластер соседних детекций ─────────────────────────────────


@dataclass(frozen=True)
class DetectionCluster:
    """Кластер соседних детекций одного физического источника (VO).

    Attributes
    ----------
    kx, ky, range_bin : координаты центроида — берутся у детекции-пика
                        (максимальный level_db среди членов кластера)
    peak_level_db     : максимум level_db среди детекций кластера
    n_members         : число исходных детекций, слившихся в кластер
    """

    kx: float
    ky: float
    range_bin: int
    peak_level_db: float
    n_members: int


# ── Кластеризатор ─────────────────────────────────────────────────────────────


class DetectionClusterer:
    """Группирует соседние `Detection` (угол + дальность) в кластеры.

    Один источник занимает несколько ячеек CFAR-сетки (мейнлоуб/страддл) —
    сливаем их в один кандидат методом single-linkage (связные компоненты
    графа близости). Не мутирует вход, детерминирован.

    Parameters
    ----------
    angle_tol : допуск по |Δkx| и |Δky| (угловые единицы оси куба)
    range_tol : допуск по |Δrange_bin| (число бинов дальности)
    """

    def __init__(self, angle_tol: float = 1.0, range_tol: int = 1) -> None:
        if angle_tol < 0.0:
            raise ValueError(f"angle_tol должно быть >= 0, получено {angle_tol}")
        if range_tol < 0:
            raise ValueError(f"range_tol должно быть >= 0, получено {range_tol}")
        self._angle_tol = angle_tol
        self._range_tol = range_tol

    @property
    def angle_tol(self) -> float:
        return self._angle_tol

    @property
    def range_tol(self) -> int:
        return self._range_tol

    def _are_neighbors(self, a: Detection, b: Detection) -> bool:
        """Связность двух детекций: близки по углу И по дальности одновременно."""
        return (
            abs(a.kx - b.kx) <= self._angle_tol
            and abs(a.ky - b.ky) <= self._angle_tol
            and abs(a.range_bin - b.range_bin) <= self._range_tol
        )

    def cluster(self, detections: list[Detection]) -> list[DetectionCluster]:
        """Группирует детекции в кластеры (single-linkage). Вход не мутируется.

        Returns
        -------
        list[DetectionCluster]
            Отсортировано по (range_bin, kx, ky) центроида — детерминировано.
        """
        if not detections:
            return []

        # Детерминированный порядок обработки (не мутирует исходный список).
        ordered = sorted(detections, key=lambda d: (d.range_bin, d.kx, d.ky))
        n = len(ordered)

        # ── union-find (path compression + union by rank) ───────────────────
        parent = list(range(n))
        rank = [0] * n

        def find(i: int) -> int:
            root = i
            while parent[root] != root:
                root = parent[root]
            while parent[i] != root:
                parent[i], i = root, parent[i]
            return root

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri == rj:
                return
            if rank[ri] < rank[rj]:
                ri, rj = rj, ri
            parent[rj] = ri
            if rank[ri] == rank[rj]:
                rank[ri] += 1

        for i in range(n):
            for j in range(i + 1, n):
                if self._are_neighbors(ordered[i], ordered[j]):
                    union(i, j)

        # ── группировка индексов по корню компоненты ─────────────────────────
        components: dict[int, list[int]] = {}
        for i in range(n):
            components.setdefault(find(i), []).append(i)

        clusters: list[DetectionCluster] = []
        for members_idx in components.values():
            members = [ordered[i] for i in members_idx]
            peak = max(members, key=lambda d: d.level_db)
            clusters.append(DetectionCluster(
                kx=peak.kx,
                ky=peak.ky,
                range_bin=peak.range_bin,
                peak_level_db=peak.level_db,
                n_members=len(members),
            ))

        clusters.sort(key=lambda c: (c.range_bin, c.kx, c.ky))
        return clusters
