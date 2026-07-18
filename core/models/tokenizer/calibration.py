"""Калибровка/валидация триажа §4.12 -- синтетический датасет {noise,source,smeared}.

Якоря `RuleBasedTriage` (§4.11) откалиброваны на ОДНОЙ модельной сцене (иллюстративны,
см. докстринг `RuleBasedTriage`). Здесь строится синтетический датасет по нескольким
апертурам и SNR, прогоняется через триаж и собирается confusion-матрица + Pfa --
экспериментальное закрытие открытого вопроса §4.12 п.2 ("пороги калибруются на
синтетическом датасете по заданной вероятности ложной тревоги").

Реюз (не плодим сущности): `FeatureExtractor` (§4.5), `RuleBasedTriage`/`SliceTriage`
(§4.11), `angular_fft` (P5, угловой FFT с паддингом до 2ⁿ), `ArrayGrid.steering`
(фазовый вектор наведения). Синтетика строится ТЕМ ЖЕ путём, что `tests/test_tokenizer.py`
(поле апертуры -> `angular_fft` -> P=|A|² -> `FeatureExtractor.extract`), но с
варьируемыми апертурой/SNR/углом вместо фиксированной модельной сцены §4.11.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...config.array_config import ArrayConfig
from ...generators.grid import ArrayGrid
from ..angular_fft import angular_fft
from .features import FeatureExtractor, FeatureVector
from .triage import NOISE, SMEARED, SOURCE, RuleBasedTriage, SliceTriage

_Triple = tuple[float, float, float]


@dataclass(frozen=True)
class ClassStats:
    """Сводная статистика признаков одного класса (медиана, p10, p90) -- Value Object."""

    label: str
    n: int
    pr: _Triple
    hoyer: _Triple
    main_frac: _Triple
    lobe_ratio: _Triple


class TriageCalibrator:
    """Строит синтетический датасет {noise,source,smeared} и валидирует `RuleBasedTriage`.

    Parameters
    ----------
    apertures    : список апертур (nx, ny), по которым варьируется датасет (проверка
                   инвариантности к M, §4.12 открытый вопрос, F9).
    snr_db_list  : список SNR (дБ по амплитуде источника относительно sigma=1 шума
                   на элемент решётки) для класса source/smeared.
    n_per_class  : число сэмплов на класс на КАЖДУЮ комбинацию (апертура, SNR).
    seed         : корневой seed (детерминизм, `np.random.default_rng`).
    """

    _NOISE_SIGMA = 1.0
    _N_SMEARED_COMPONENTS = 10

    def __init__(self, apertures: list[tuple[int, int]], snr_db_list: list[float],
                 n_per_class: int = 25, seed: int = 0) -> None:
        if n_per_class < 1:
            raise ValueError(f"n_per_class должен быть >= 1, получено {n_per_class}")
        self._apertures = tuple(apertures)
        self._snr_db_list = tuple(snr_db_list)
        self._n_per_class = n_per_class
        self._seed = seed
        self._extractor = FeatureExtractor()
        self._dataset: dict[str, list[FeatureVector]] | None = None
        self._by_aperture: dict[tuple[int, int], dict[str, list[FeatureVector]]] | None = None

    # ── построение датасета ──────────────────────────────────────────────────

    def build_dataset(self) -> dict[str, list[FeatureVector]]:
        """Датасет {noise,source,smeared} -> список `FeatureVector`, по ВСЕМ апертурам/SNR.

        Детерминировано: корневой `rng` порождает независимый seed на каждую
        комбинацию (апертура, SNR), внутри которой noise/source/smeared сэмплы
        тянутся из одного потока (тот же seed => тот же датасет).
        """
        if self._dataset is not None:
            return self._dataset

        combined: dict[str, list[FeatureVector]] = {NOISE: [], SOURCE: [], SMEARED: []}
        by_aperture: dict[tuple[int, int], dict[str, list[FeatureVector]]] = {}
        root_rng = np.random.default_rng(self._seed)

        for nx, ny in self._apertures:
            grid = ArrayGrid.from_config(ArrayConfig(nx, ny))
            per_aperture: dict[str, list[FeatureVector]] = {NOISE: [], SOURCE: [], SMEARED: []}

            for snr_db in self._snr_db_list:
                combo_seed = int(root_rng.integers(0, 2**31 - 1))
                rng = np.random.default_rng(combo_seed)
                for _ in range(self._n_per_class):
                    f_noise = self._extractor.extract(self._noise_map(nx, ny, rng))
                    f_source = self._extractor.extract(self._source_map(grid, nx, ny, snr_db, rng))
                    f_smeared = self._extractor.extract(self._smeared_map(grid, nx, ny, snr_db, rng))
                    per_aperture[NOISE].append(f_noise)
                    per_aperture[SOURCE].append(f_source)
                    per_aperture[SMEARED].append(f_smeared)

            by_aperture[(nx, ny)] = per_aperture
            for label in (NOISE, SOURCE, SMEARED):
                combined[label].extend(per_aperture[label])

        self._dataset = combined
        self._by_aperture = by_aperture
        return combined

    def dataset_by_aperture(self) -> dict[tuple[int, int], dict[str, list[FeatureVector]]]:
        """Датасет, разбитый по апертуре -- для проверки инвариантности к M."""
        self.build_dataset()
        assert self._by_aperture is not None
        return self._by_aperture

    # ── синтетические угловые карты (реюз angular_fft/ArrayGrid.steering) ───

    def _noise_field(self, nx: int, ny: int, rng: np.random.Generator) -> np.ndarray:
        return (rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))) \
            * self._NOISE_SIGMA / np.sqrt(2.0)

    @staticmethod
    def _power(field: np.ndarray) -> np.ndarray:
        """P = |angular_fft(field)|² на единственном срезе дальности (без окна апертуры)."""
        cube3 = field[:, :, None].astype(np.complex128)
        spectrum = angular_fft(cube3)
        a = spectrum[:, :, 0]
        return a.real ** 2 + a.imag ** 2

    def _noise_map(self, nx: int, ny: int, rng: np.random.Generator) -> np.ndarray:
        return self._power(self._noise_field(nx, ny, rng))

    def _source_map(self, grid: ArrayGrid, nx: int, ny: int, snr_db: float,
                     rng: np.random.Generator) -> np.ndarray:
        """Один точечный источник: amp*steering(kx0,ky0) + шум, (kx0,ky0) случайны."""
        kx0 = float(rng.uniform(-nx / 2.0 + 1.0, nx / 2.0 - 1.0))
        ky0 = float(rng.uniform(-ny / 2.0 + 1.0, ny / 2.0 - 1.0))
        amp = 10.0 ** (snr_db / 20.0) * self._NOISE_SIGMA
        field = amp * grid.steering(kx0, ky0) + self._noise_field(nx, ny, rng)
        return self._power(field)

    def _smeared_map(self, grid: ArrayGrid, nx: int, ny: int, snr_db: float,
                      rng: np.random.Generator) -> np.ndarray:
        """Широкоугольный "заградительный" источник: сумма steering под неск. углами.

        Разброс углов (`spread_x/y`) масштабируется с апертурой -- фиксированный
        ФИЗИЧЕСКИЙ угловой разброс при более тонком разрешении большей решётки
        занимает пропорционально больше угловых бинов.
        """
        base_kx = float(rng.uniform(-nx / 2.0 + 2.0, nx / 2.0 - 2.0))
        base_ky = float(rng.uniform(-ny / 2.0 + 2.0, ny / 2.0 - 2.0))
        amp = 10.0 ** (snr_db / 20.0) * self._NOISE_SIGMA
        spread_x = 3.5 * (nx / 16.0)
        spread_y = 3.5 * (ny / 16.0)

        field = self._noise_field(nx, ny, rng)
        for _ in range(self._N_SMEARED_COMPONENTS):
            dkx = float(rng.uniform(-spread_x, spread_x))
            dky = float(rng.uniform(-spread_y, spread_y))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            comp_amp = amp * float(rng.uniform(0.5, 1.0)) / np.sqrt(self._N_SMEARED_COMPONENTS)
            field = field + comp_amp * np.exp(1j * phase) * grid.steering(base_kx + dkx, base_ky + dky)
        return self._power(field)

    # ── статистика и валидация ───────────────────────────────────────────────

    def class_stats(self) -> list[ClassStats]:
        """Медиана/p10/p90 признаков по каждому классу (по всему датасету)."""
        dataset = self.build_dataset()
        stats: list[ClassStats] = []
        for label in (SOURCE, NOISE, SMEARED):
            vectors = dataset[label]
            pr = np.array([v.pr for v in vectors])
            hoyer = np.array([v.hoyer for v in vectors])
            main_frac = np.array([v.main_frac for v in vectors])
            lobe_ratio = np.array([v.lobe_ratio for v in vectors])
            stats.append(ClassStats(
                label=label, n=len(vectors),
                pr=self._triple(pr), hoyer=self._triple(hoyer),
                main_frac=self._triple(main_frac), lobe_ratio=self._triple(lobe_ratio),
            ))
        return stats

    @staticmethod
    def _triple(arr: np.ndarray) -> _Triple:
        return (
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 10)),
            float(np.percentile(arr, 90)),
        )

    def validate(self, triage: SliceTriage | None = None) -> dict:
        """Confusion-матрица + accuracy по классам + Pfa шума, по ВСЕМУ датасету.

        `pfa_noise` -- доля истинно-шумовых сэмплов, классифицированных НЕ как noise
        (т.е. ложно "продетектированных" как source/smeared) -- вероятность ложной
        тревоги триажа на чистом шуме (§4.12 п.2).
        """
        dataset = self.build_dataset()
        return self._validate_dataset(dataset, triage or RuleBasedTriage())

    def validate_by_aperture(self, triage: SliceTriage | None = None) -> dict[tuple[int, int], dict]:
        """`validate()`, но отдельно на каждую апертуру -- проверка инвариантности к M."""
        triage = triage or RuleBasedTriage()
        return {ap: self._validate_dataset(ds, triage) for ap, ds in self.dataset_by_aperture().items()}

    @staticmethod
    def _validate_dataset(dataset: dict[str, list[FeatureVector]], triage: SliceTriage) -> dict:
        confusion: dict[str, dict[str, int]] = {
            true_label: {NOISE: 0, SOURCE: 0, SMEARED: 0} for true_label in (NOISE, SOURCE, SMEARED)
        }
        for true_label, vectors in dataset.items():
            for f in vectors:
                pred, _score = triage.classify(f)
                confusion[true_label][pred] = confusion[true_label].get(pred, 0) + 1

        def accuracy(label: str) -> float:
            row = confusion[label]
            total = sum(row.values())
            return row.get(label, 0) / total if total else 0.0

        noise_row = confusion[NOISE]
        total_noise = sum(noise_row.values())
        pfa_noise = ((total_noise - noise_row.get(NOISE, 0)) / total_noise) if total_noise else 0.0

        return {
            "confusion": confusion,
            "accuracy": {
                SOURCE: accuracy(SOURCE),
                NOISE: accuracy(NOISE),
                SMEARED: accuracy(SMEARED),
            },
            "pfa_noise": pfa_noise,
        }
