"""VolumeTokenizer -- Template Method (гл.4 §4.2-4.9, гл.4-бис §4-бис.3, TASK §3-4).

Одно ядро для плоского (`window_l=1`, гл.4) и объёмного (`window_l>1`, гл.4-бис)
случая: под-окно `nx x ny x window_l` -> признаки -> триаж -> (если не шум) поиск
пиков -> `SliceToken`. Куб читается по ссылке, НЕ мутируется.

Проход 2 (`assemble_range`) работает уже над потоком `SliceToken` (не по кубу) --
дёшево (гл.4 §4.9): собирает `source`-токены под одним углом по дальности в
`target`/`comb`/`barrage`.
"""
from __future__ import annotations

import numpy as np

from ..result import SpectralCube
from .cfar import OsCfarDetector
from .features import FeatureExtractor
from .tokens import BARRAGE, COMB, TARGET, PeakInfo, RangeVerdict, SliceToken
from .triage import NOISE, SOURCE, RuleBasedTriage, SliceTriage


class VolumeTokenizer:
    """Template Method: `tokenize(cube) -> list[SliceToken]`.

    Parameters
    ----------
    window_l : глубина окна по дальности (`L`). `L=1` -- плоский случай гл.4,
               частный случай `L>1` (гл.4-бис §4-бис.1 -- "одно ядро, много интерфейсов").
    step     : шаг окна вдоль дальности. По умолчанию 1 (скользящее окно без пропусков --
               корректность важнее скорости на этом этапе, см. приоритеты проекта).
    extractor, cfar, triage : сменные стратегии (DI, Composition Root связывает снаружи).
    """

    def __init__(
        self,
        window_l: int = 1,
        step: int = 1,
        extractor: FeatureExtractor | None = None,
        cfar: OsCfarDetector | None = None,
        triage: SliceTriage | None = None,
    ) -> None:
        if window_l < 1:
            raise ValueError(f"window_l должен быть >= 1, получено {window_l}")
        if step < 1:
            raise ValueError(f"step должен быть >= 1, получено {step}")
        self._l = window_l
        self._step = step
        self._extractor = extractor or FeatureExtractor()
        self._cfar = cfar or OsCfarDetector()
        self._triage = triage or RuleBasedTriage()

    # ── Template Method ──────────────────────────────────────────────────────

    def tokenize(self, cube: SpectralCube) -> list[SliceToken]:
        """Проход 1: список `SliceToken` (шумовые окна не пишутся -- разрежённый выход)."""
        mag = cube.magnitude              # только чтение, не мутируется
        _, _, n_range = mag.shape
        window_l = self._l

        tokens: list[SliceToken] = []
        r0 = 0
        while r0 + window_l <= n_range:
            window_mag = mag[:, :, r0:r0 + window_l]
            power = window_mag.astype(np.float64) ** 2       # P = |A|² (E7 -- вход extractor'а)

            f = self._extractor.extract(power)
            label, score = self._triage.classify(f)

            if label != NOISE:                                # пустые/шумовые -- не пишем (§4.7)
                peaks = self._find_peaks(cube, power, r0, window_l, n_range)
                tokens.append(SliceToken(r=r0, peaks=tuple(peaks), f=f, label=label, score=score))

            r0 += self._step
        return tokens

    # ── шаг 4: поиск пиков + кромка ──────────────────────────────────────────

    def _find_peaks(
        self, cube: SpectralCube, power: np.ndarray, r0: int, window_l: int, n_range: int,
    ) -> list[PeakInfo]:
        # Пики -- по углу (kx,ky), TASK §2: без k_z. Окно по глубине сводим max'ом
        # (аналог SquareView.reduce_square(reduce_mode="max") -- пик может быть в любой
        # плоскости окна, max по глубине не теряет его положение по углу).
        power2d = power.max(axis=2) if window_l > 1 else power[:, :, 0]
        raw_peaks = self._cfar.find_peaks(power2d)

        mag = cube.magnitude
        peaks: list[PeakInfo] = []
        for (ix, iy), amp_power in raw_peaks:
            kx_val = float(cube.kx.values[ix])
            ky_val = float(cube.ky.values[iy])
            amp = float(np.sqrt(max(amp_power, 0.0)))          # обратно в амплитуду |A|
            left = float(mag[ix, iy, r0 - 1]) if r0 - 1 >= 0 else 0.0
            right = float(mag[ix, iy, r0 + window_l]) if r0 + window_l < n_range else 0.0
            edge = right - left                                # кромка: нарастание/спад (§4.6)
            peaks.append(PeakInfo(kx=kx_val, ky=ky_val, amp=amp, edge=edge))
        return peaks


# ── Проход 2 (гл.4 §4.9) ─────────────────────────────────────────────────────

def assemble_range(
    tokens: list[SliceToken],
    min_barrage_run: int = 3,
    period_confirm_ratio: float = 0.99,
) -> list[RangeVerdict]:
    """Сборка `source`-токенов по дальности под одним углом -> `target`/`comb`/`barrage`.

    Работает по единицам токенов (не по кубу) -- дёшево (§4.9). `smeared`/`noise`
    токены в сборке не участвуют -- проход 2 различает только среди `source`
    (гл.4 §4.9: "Берём токены с меткой «источник»").

    Parameters
    ----------
    min_barrage_run      : минимальная длина сплошного (без пропусков) участка `r`,
                            чтобы считать группу `barrage` (иначе -- совпадение 2 target).
    period_confirm_ratio : порог подтверждения регулярности автокорреляцией индикатора
                            присутствия токена по `r` (см. `_dominant_period`).
    """
    groups: dict[tuple[float, float], set[int]] = {}
    for tok in tokens:
        if tok.label != SOURCE:
            continue
        for peak in tok.peaks:
            key = (round(peak.kx, 6), round(peak.ky, 6))
            groups.setdefault(key, set()).add(tok.r)

    verdicts: list[RangeVerdict] = []
    for (kx, ky), r_set in groups.items():
        r_list = sorted(r_set)
        kind, lead_r, period = _classify_group(r_list, min_barrage_run, period_confirm_ratio)
        verdicts.append(RangeVerdict(kx=kx, ky=ky, kind=kind, lead_r=lead_r, period_dr=period))
    return verdicts


def _classify_group(
    r_list: list[int], min_barrage_run: int, period_confirm_ratio: float,
) -> tuple[str, int, float | None]:
    n = len(r_list)
    if n == 1:
        return TARGET, r_list[0], None

    lo, hi = r_list[0], r_list[-1]
    span = hi - lo + 1
    contiguous = span == n

    if contiguous and span >= min_barrage_run:
        return BARRAGE, lo, None

    period = _dominant_period(r_list, period_confirm_ratio)
    if period is not None:
        return COMB, lo, period

    if contiguous:
        return BARRAGE, lo, None

    # Нерегулярная структура (не одиночка, не сплошь, не подтверждённый период) --
    # открытый вопрос гл.4 §4.12 (рваная гребёнка -> глубокий уровень LSTM, гл.7,
    # вне P1). Консервативный fallback: помечаем передний край кандидатом, тип
    # оставляем "target" (не выдумываем несуществующий 4-й вид на этом уровне).
    return TARGET, lo, None


def _dominant_period(r_list: list[int], confirm_ratio: float) -> float | None:
    """Регулярный шаг Δr между токенами группы, подтверждённый автокорреляцией (§4.9)."""
    diffs = np.diff(np.asarray(r_list, dtype=np.int64))
    if diffs.size == 0 or diffs[0] <= 0 or not np.all(diffs == diffs[0]):
        return None
    period = float(diffs[0])

    lo, hi = r_list[0], r_list[-1]
    span = hi - lo + 1
    indicator = np.zeros(span, dtype=np.float64)
    for r in r_list:
        indicator[r - lo] = 1.0

    lag = int(round(period))
    if lag <= 0 or lag >= span:
        return period  # цепочка коротка для автокорр-проверки -- доверяем точному diff'у

    num_pairs = span - lag
    matched = float(np.sum(indicator[:num_pairs] * indicator[lag:lag + num_pairs]))
    energy = float(np.sum(indicator[:num_pairs]))
    if energy <= 0.0 or matched / energy < confirm_ratio:
        return None
    return period
