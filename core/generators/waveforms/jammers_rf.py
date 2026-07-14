"""Время-доменные помехи (Strategy, §5 taska P5) — патент §1.2 + промышленные (industrial §5-6).

Каждая помеха — `Waveform`-подкласс, шаг 1 (сырой 1D-сигнал) свой, шаги 2-5
(окно → раскладка n×n через steering → шум по SNR → упаковка в `SignalField`)
реюзают `_pipeline.render_pipeline` (тот же приём, что `am.py`/`fm.py`/`phase_code.py`).

Куб-уровневые `DrfmComb`/`BarrageJammer`/`HamEmitter` (`core/generators/jammers.py`) —
НЕ трогаем, взяты только как конвенция параметров (kx/ky/lead/spacing/count/power).

Калибровка мощности (J1) — исключительно через существующий `spec.snr_db` +
`amplitude_for_snr(spec)` (та же формула `A=√(σ²·10^(SNR/10))`, тот же эталон
`σ²=NOISE_POWER`), никакого второго `jnr_db` в meta.

Вся случайность — через переданный `rng` (R6): `rng.standard_normal`, `rng.poisson`,
`rng.uniform`, а для α-stable — `scipy.stats.levy_stable.rvs(..., random_state=rng)`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._pipeline import amplitude_for_snr, render_pipeline
from .base import Waveform, WaveformSpec
from .field import Modulation, SignalField
from .reference import cw_numpy, getX_numpy

if TYPE_CHECKING:
    from ..backends.base import GenBackend

# ── BarrageRfJammer (J4: когерентный/направленный, не диффузный) ───────────────

DEFAULT_BARRAGE_AMPLITUDE: float = 1.0


def _barrage_numpy(length: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """`s(t) = amplitude · white_complex_noise(n)` — unit-variance комплексный белый шум.

    `white_complex_noise` — CN(0,1) (I/Q по 0.5 каждая), домноженный на `amplitude`
    даёт среднюю мощность `amplitude**2` — та же шкала, что `amplitude_for_snr`
    для детерминированных тонов (|exp(j·)|=1 → средняя мощность = amplitude**2).
    """
    noise = (rng.standard_normal(length) + 1j * rng.standard_normal(length)) / np.sqrt(2.0)
    return (amplitude * noise).astype(np.complex64)


class BarrageRfJammer(Waveform):
    """Заградительная помеха: широкополосный шум с одного угла (J4 — коэрентно, через steering).

    `render_pipeline` домножает ОДИН 1D-шум на steering-вектор → массив rank-1
    (одинаковая по модулю картина на всех элементах, разная только фаза наведения) —
    это и есть «направленный» barrage, диффузный (независимый шум на элемент) через
    этот пайплайн не получить (уже даёт `add_noise` тепловой пол).
    """

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        signal = _barrage_numpy(spec.n_samples, amplitude, rng)
        return render_pipeline(backend, spec, rng, signal, Modulation.BARRAGE)


# ── SmspJammer (размытие спектра — DRFM пересобирает подчирпами) ───────────────

DEFAULT_K_SEGMENTS: int = 8   # число подчирпов (K), делит n_samples нацело при baseline 8192


def _smsp_numpy(fs: float, length: int, f0: float, amplitude: float,
                 fdev: float, k_segments: int) -> np.ndarray:
    """K подчирпов, каждый свипует ВСЮ полосу `fdev` за `1/K` длины окна (μ_smsp=K·μ).

    Каждый сегмент — `reference.getX_numpy` (та же формула зонда) на укороченном
    `seg_len = length // k_segments` — состыкованы подряд. Хвост `length % k_segments`
    (если есть) остаётся нулевым (энергия вне последнего целого сегмента = 0, не
    портит центрально-симметричную формулу getX на каждом сегменте).
    """
    seg_len = length // k_segments
    out = np.zeros(length, dtype=np.complex64)
    for k in range(k_segments):
        seg = getX_numpy(fs, seg_len, f0, amplitude, 0.0, fdev, 1.0)
        out[k * seg_len:(k + 1) * seg_len] = seg
    return out


class SmspJammer(Waveform):
    """SMSP: K подчирпов на всю полосу за укороченное время → размазанный спектр после дечирпа.

    Сырой спектр SMSP и обычного ЛЧМ той же ширины полосы почти неразличимы (оба
    занимают `fdev`) — признак виден только ПОСЛЕ дечирпа опорным ЛЧМ (J2, тест —
    `tests/test_generators.py`): matched ЛЧМ → острый пик, SMSP → размазан/пики.
    """

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        k_segments = int(spec.meta.get("k_segments", DEFAULT_K_SEGMENTS))
        signal = _smsp_numpy(spec.fs, spec.n_samples, spec.carrier_hz, amplitude,
                              spec.fdev_hz, k_segments)
        return render_pipeline(backend, spec, rng, signal, Modulation.SMSP)


# ── DrfmRepeaterJammer (гребёнка ложных целей — задержанные копии опорного ЛЧМ) ─

DEFAULT_LEAD_S: float = 1e-6      # задержка первой копии, с
DEFAULT_SPACING_S: float = 1e-6   # шаг между копиями, с
DEFAULT_COUNT: int = 5            # число ложных целей
DEFAULT_DECAY: float = 0.85       # затухание амплитуды копии к копии


def _drfm_repeater_numpy(fs: float, length: int, f0: float, amplitude: float, fdev: float,
                          lead_s: float, spacing_s: float, count: int, decay: float) -> np.ndarray:
    """`Σ_i a·decay^i · lfm_ref(t − τ_i)`, `τ_i = lead + i·spacing` (сек, → отсчёты round(τ_i·fs)).

    Опорный ЛЧМ — `reference.getX_numpy` (та же формула зонда, без задержки/окна,
    `amplitude=1.0`); копии за пределом окна (`shift >= length`) отбрасываются.
    """
    ref = getX_numpy(fs, length, f0, 1.0, 0.0, fdev, 1.0)
    out = np.zeros(length, dtype=np.complex64)
    for i in range(count):
        tau = lead_s + i * spacing_s
        shift = round(tau * fs)
        if shift >= length:
            continue
        amp_i = amplitude * (decay ** i)
        out[shift:] += (amp_i * ref[:length - shift]).astype(np.complex64)
    return out


class DrfmRepeaterJammer(Waveform):
    """DRFM-ретранслятор: `count` задержанных затухающих копий опорного ЛЧМ (τ_i=lead+i·spacing)."""

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        lead_s = float(spec.meta.get("lead_s", DEFAULT_LEAD_S))
        spacing_s = float(spec.meta.get("spacing_s", DEFAULT_SPACING_S))
        count = int(spec.meta.get("count", DEFAULT_COUNT))
        decay = float(spec.meta.get("decay", DEFAULT_DECAY))
        signal = _drfm_repeater_numpy(spec.fs, spec.n_samples, spec.carrier_hz, amplitude,
                                       spec.fdev_hz, lead_s, spacing_s, count, decay)
        return render_pipeline(backend, spec, rng, signal, Modulation.DRFM_REPEATER)


# ── IndustrialCwJammer (INT_CW, 🔴1 — CW чужого радара) ─────────────────────────

# J3: маппинг — f_int держим как ДОЛЮ fs (не абсолютное число), чтобы при любом
# baseline fs остаться < fs/2 (Найквист). При fs=12МГц → f_int≈3.48МГц (< fs/2=6МГц,
# заметно правее несущей f0=2МГц, острый пик легко отличим от рабочей полосы ЛЧМ).
DEFAULT_F_INT_FRACTION: float = 0.29


class IndustrialCwJammer(Waveform):
    """CW чужого радара: `A·exp(j·2π·f_int·t+φ)` — острый пик, имитатор точечной цели.

    Реюз `reference.cw_numpy`. `f_int` — `spec.meta["f_int"]`, дефолт `fs*0.29`
    (baseband-модель приёмника, J3 — не «настоящая» РЧ несущая, см. отчёт P5).
    """

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        f_int = float(spec.meta.get("f_int", spec.fs * DEFAULT_F_INT_FRACTION))
        signal = cw_numpy(spec.fs, spec.n_samples, f_int, amplitude, spec.phase)
        return render_pipeline(backend, spec, rng, signal, Modulation.INDUSTRIAL_CW)


# ── ImpulsiveArcJammer (IMP_ARC, 🔴2 — сварочная дуга/разряд) ───────────────────

DEFAULT_LAMBDA_HZ: float = 5e4       # интенсивность пуассоновского потока импульсов, 1/с
DEFAULT_TAU_DECAY_S: float = 3e-7    # затухающий хвост импульса, с (industrial §4.2: 0.1-1 мкс)
DEFAULT_ALPHA_STABLE: float = 1.4    # индекс α-stable для амплитуд r_k


def _impulsive_arc_numpy(fs: float, length: int, amplitude: float, rng: np.random.Generator,
                          lam_hz: float, tau_decay_s: float, alpha: float) -> np.ndarray:
    """`s(t) = Σ_k A_k·δ(t−t_k) * h(t)` — J5: СВЁРТКА `δ*h`, не буквальное `δ·exp`.

    `t_k` — пуассоновский поток (интенсивность `lam_hz`), `A_k=r_k·exp(jφ_k)` (J6):
    `r_k` ~ |α-stable| (`scipy.stats.levy_stable`, генерится пачкой; фолбэк —
    тяжелохвостый Стьюдент `df≈2.2`, если scipy недоступен), `φ_k`~U(0,2π).
    `h(u)=exp(−u/τ)·[u≥0]` — причинный затухающий импульс, применяется через
    `np.convolve` (честная свёртка, хвост импульса после δ, не одиночный отсчёт).
    """
    dt = 1.0 / fs
    duration = length * dt
    n_events = int(rng.poisson(lam_hz * duration))
    if n_events == 0:
        return np.zeros(length, dtype=np.complex64)

    t_events = np.sort(rng.uniform(0.0, duration, size=n_events))
    idx = np.clip((t_events * fs).astype(np.int64), 0, length - 1)

    try:
        from scipy.stats import levy_stable
        r = np.abs(levy_stable.rvs(alpha, 0.0, size=n_events, random_state=rng))
    except ImportError:
        # фолбэк (spec §подводные камни): тяжелохвостое распределение без scipy.
        r = np.abs(rng.standard_t(df=2.2, size=n_events))

    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_events)
    amps = r * np.exp(1j * phi)

    delta = np.zeros(length, dtype=np.complex128)
    np.add.at(delta, idx, amps)

    kernel_len = max(1, min(length, int(round(10.0 * tau_decay_s * fs)) + 1))
    u = np.arange(kernel_len) * dt
    h = np.exp(-u / tau_decay_s)

    conv = np.convolve(delta, h, mode="full")[:length]
    return (amplitude * conv).astype(np.complex64)


class ImpulsiveArcJammer(Waveform):
    """Сварочная дуга/разряд: пуассоновский поток импульсов с тяжёлыми α-stable хвостами.

    `lambda_hz` (интенсивность, 1/с), `tau_decay_s` (постоянная затухания хвоста),
    `alpha_stable` (индекс α-stable) — `spec.meta`. Высокий эксцесс + разреженность
    во времени (см. тест `imp_arc.kurtosis_and_sparsity`).
    """

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        lam_hz = float(spec.meta.get("lambda_hz", DEFAULT_LAMBDA_HZ))
        tau_decay_s = float(spec.meta.get("tau_decay_s", DEFAULT_TAU_DECAY_S))
        alpha = float(spec.meta.get("alpha_stable", DEFAULT_ALPHA_STABLE))
        signal = _impulsive_arc_numpy(spec.fs, spec.n_samples, amplitude, rng,
                                       lam_hz, tau_decay_s, alpha)
        return render_pipeline(backend, spec, rng, signal, Modulation.IMPULSIVE_ARC)


# ── VfdHarmonicJammer (HAR_VFD, 🔴3 — гармоники VFD/IGBT) ───────────────────────

# J3: маппинг гребёнки — оставляем НЧ (около DC), НЕ переносим к несущей f0.
# Обоснование (задокументировано в отчёте P5): VFD/IGBT — кондуктивная помеха,
# физически НЕ проходит через смеситель приёмника на несущую f0, попадает в тракт
# напрямую (как сетевая наводка) — маппинг «около f0» был бы нефизичен. Все
# `n·f_sw` обязаны быть `< fs/2` — цикл ниже обрывается, как только гармоника
# вышла за Найквист.
DEFAULT_F_SW_HZ: float = 8e3          # несущая частота IGBT-переключения (2-16 кГц), середина
DEFAULT_N_HARMONICS: int = 15
DEFAULT_DECAY_EXPONENT: float = 1.0   # A_n ~ amplitude / n**decay_exponent
DEFAULT_BROADBAND_FRACTION: float = 0.1   # доля amplitude на широкополосную компоненту


def _vfd_numpy(fs: float, length: int, amplitude: float, f_sw: float, n_harmonics: int,
                decay_exponent: float, broadband_frac: float, rng: np.random.Generator) -> np.ndarray:
    """`Σ_{n=1}^{N} A_n·exp(j·(2π·n·f_sw·t+φ_n)) + широкополосная компонента` (§6.3 industrial).

    `A_n = amplitude / n**decay_exponent`; гармоники `n·f_sw >= fs/2` не добавляются
    (J3 — Найквист). Комплексная baseband-запись (как у остальных волн), не буквальный
    `cos` — тот же приём, что CW/АМ/ЧМ в этом пакете.
    """
    t = np.arange(length, dtype=np.float64) / fs
    signal = np.zeros(length, dtype=np.complex128)
    for n in range(1, n_harmonics + 1):
        f_n = n * f_sw
        if f_n >= fs / 2.0:
            break
        a_n = amplitude / (n ** decay_exponent)
        phi_n = rng.uniform(0.0, 2.0 * np.pi)
        signal += a_n * np.exp(1j * (2.0 * np.pi * f_n * t + phi_n))

    bw_power = (amplitude * broadband_frac) ** 2
    scale = np.sqrt(bw_power / 2.0)
    signal += scale * (rng.standard_normal(length) + 1j * rng.standard_normal(length))
    return signal.astype(np.complex64)


class VfdHarmonicJammer(Waveform):
    """VFD/IGBT: гармонический гребень `n·f_sw` + широкополосная компонента (industrial §4.1).

    `f_sw` (2-16 кГц), `n_harmonics`, `decay_exponent`, `broadband_frac` — `spec.meta`.
    """

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        f_sw = float(spec.meta.get("f_sw", DEFAULT_F_SW_HZ))
        n_harmonics = int(spec.meta.get("n_harmonics", DEFAULT_N_HARMONICS))
        decay_exponent = float(spec.meta.get("decay_exponent", DEFAULT_DECAY_EXPONENT))
        broadband_frac = float(spec.meta.get("broadband_frac", DEFAULT_BROADBAND_FRACTION))
        signal = _vfd_numpy(spec.fs, spec.n_samples, amplitude, f_sw, n_harmonics,
                             decay_exponent, broadband_frac, rng)
        return render_pipeline(backend, spec, rng, signal, Modulation.VFD_HARMONIC)
