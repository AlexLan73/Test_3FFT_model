"""PhaseCodeWaveform — ФМн-зонд (BPSK через M-послед. ±1), §5/§6.2 спеки, таск P4.

`s(t) = amplitude · code(t) · exp(j·(2π·f0·t + phase))` — BPSK **напрямую через
±1 код** (эквивалент `exp(jπ·c)` с `c∈{0,1}`, но без перевода). `code(t)` — чип
±1 по времени, `L` чипов `m_sequence(degree,...)` растянуты на `n_samples`
(`chip = floor(k·L/n_samples)`). `degree`/`seed`/`poly_taps` — из `spec.meta`
(G10), дефолты из `mseq`.

🔴 H1 (`TASK_signal_generators_p4.md`): `render()` отдаёт **комплексный
passband** `SignalField` (для спектра/датасета) — готовый GPU-коррелятор
(`FMCorrelatorROCm`) с ним НЕ работает. Он берёт **реальный ±1 код baseband**
(`m_sequence(...)`), который мы кладём отдельно в `field.meta["code"]`
(float32, длина `2^degree-1`, ДО растяжения на `n_samples` и ДО несущей) —
именно его подают в `prepare_reference_from_data`/`process`, а не `field.data`.
`complex_array.astype(np.float32)` на комплексном массиве молча роняет мнимую
часть (ComplexWarning) — так делать нельзя, поэтому код формируется real
float32 с самого начала (в `mseq.m_sequence`), отдельно от `s(t)`.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

from ._pipeline import amplitude_for_snr, render_pipeline
from .base import Waveform, WaveformSpec
from .field import Modulation, SignalField
from .mseq import DEFAULT_DEGREE, DEFAULT_SEED, m_sequence

if TYPE_CHECKING:
    from ..backends.base import GenBackend


def _phase_code_numpy(fs: float, length: int, f0: float, amplitude: float,
                       phase: float, code: np.ndarray) -> np.ndarray:
    """1D ФМн-сигнал (шаг 1 пайплайна §4.0): код ±1 растянут на `length` отсчётов,
    домножен на несущую. Не мутирует `code`."""
    t = np.arange(length, dtype=np.float64) / fs
    l_code = code.shape[0]
    chip_idx = (np.arange(length, dtype=np.int64) * l_code) // length
    code_stretched = code[chip_idx]
    carrier = np.exp(1j * (2.0 * np.pi * f0 * t + phase))
    return (amplitude * code_stretched * carrier).astype(np.complex64)


class PhaseCodeWaveform(Waveform):
    """ФМн: `mseq.m_sequence` (код) → BPSK → окно → n×n → шум (§4.0 пайплайн).

    Параметры кода — `spec.meta`: `degree` (int, default `mseq.DEFAULT_DEGREE`=13),
    `seed` (int, default 1), `poly_taps` (tuple[int,...], default — встроенная таблица).
    """

    def render(self, backend: GenBackend, spec: WaveformSpec, rng: np.random.Generator) -> SignalField:
        amplitude = amplitude_for_snr(spec)
        degree = int(spec.meta.get("degree", DEFAULT_DEGREE))
        seed = int(spec.meta.get("seed", DEFAULT_SEED))
        # `WaveformSpec.meta`/`SignalField.meta` типизированы `Mapping[str, float]` (G10 —
        # общий случай: числовые доп.параметры). `poly_taps`/`code` — документированное
        # исключение (H1/G10-таск P4: код-массив и таплы отводов тоже кладём в meta) —
        # тип шире заявленного, поэтому явные `type: ignore` ниже.
        poly_taps_raw: Any = spec.meta.get("poly_taps")
        poly_taps = tuple(int(t) for t in poly_taps_raw) if poly_taps_raw is not None else None

        code = m_sequence(degree, seed, poly_taps)   # H1: сырой real ±1 код (до несущей)
        signal = _phase_code_numpy(spec.fs, spec.n_samples, spec.carrier_hz,
                                    amplitude, spec.phase, code)
        field = render_pipeline(backend, spec, rng, signal, Modulation.PHASE_CODE)
        # H1: код кладём в meta отдельно (VO неизменяем → пересобираем через dataclasses.replace,
        # не мутируем `field`). Коррелятору подаём field.meta["code"], НЕ field.data.
        new_meta: dict[str, float] = {**field.meta, "code": code}  # type: ignore[dict-item]
        return dataclasses.replace(field, meta=new_meta)
