"""M-последовательность (LFSR, наша) — §5/§6.2 спеки, таск P4.

Максимальной длины (2^degree − 1) псевдослучайный код ±1 через Galois LFSR.
Алгоритм **бит-в-бит идентичен** CPU-генератору готового GPU-коррелятора
(`dsp::radar::FMCorrelator::GenerateMSequence`, `DSP-GPU/radar/src/fm_correlator/
fm_correlator.cpp:46-58`): 32-битный регистр, выходной символ — старший бит
(bit31) **до** сдвига, при выходном бите=1 после сдвига делаем XOR с полиномом
(Galois-форма). Мы используем **тот же 32-битный регистр**, но полином/сид
выравниваем по старшим битам (`<< (32 - degree)`) — тогда младшие
`32-degree` бит регистра остаются нулевыми весь цикл (обратная связь ни разу
их не трогает, т.к. полином там тоже нулевой), и получается ровно
degree-битный Galois LFSR со старшим битом как output — тот же алгоритм,
другая ширина. Совпадение с `FMCorrelatorROCm.generate_msequence()` при
одинаковом (выровненном) полиноме/сиде проверено бит-в-бит (H2-тест, GPU).

Таблица `_PRIMITIVE_TAPS` — полиномы degree 7..16, **каждый программно
проверен** (полный период 2^degree−1, без более коротких циклов) — не
canonical-таблица из учебника, а самостоятельно подобранные и верифицированные
трёх-/двучлены. Ловится тестом автокорреляции (thumbtack), если ошибка.
"""
from __future__ import annotations

import numpy as np

DEFAULT_DEGREE: int = 13   # 2^13-1 = 8191 чипов — вплотную под baseline n_samples=8192 (§5.1)
DEFAULT_SEED: int = 1

# Отводы (exponents) обратной связи, БЕЗ ведущего члена x^degree и БЕЗ константы x^0
# (бит0 добавляется всегда — иначе полином не может быть примитивным). Проверено
# программно: полный период 2^degree-1 (см. отчёт P4 / MemoryBank).
_PRIMITIVE_TAPS: dict[int, tuple[int, ...]] = {
    7: (6,),
    8: (6, 5, 4),
    9: (5,),
    10: (7,),
    11: (9,),
    12: (6, 4, 1),
    13: (4, 3, 1),
    14: (5, 3, 1),
    15: (14,),
    16: (15, 13, 4),
}

_REGISTER_BITS: int = 32
_MASK32: int = 0xFFFFFFFF


def _taps_mask(degree: int, poly_taps: tuple[int, ...] | None) -> int:
    """degree-битная маска обратной связи (бит0 — константа — всегда установлен)."""
    taps = poly_taps if poly_taps is not None else _PRIMITIVE_TAPS.get(degree)
    if taps is None:
        raise ValueError(
            f"нет встроенного примитивного полинома для degree={degree} "
            f"(есть {sorted(_PRIMITIVE_TAPS)}); передай poly_taps явно"
        )
    mask = 1  # x^0 — константный член, обязателен для примитивности
    for t in taps:
        if not (0 < t < degree):
            raise ValueError(f"tap={t} должен быть в диапазоне (0, degree={degree})")
        mask |= 1 << t
    return mask


def gpu_lfsr_params(
    degree: int, seed: int = DEFAULT_SEED, poly_taps: tuple[int, ...] | None = None
) -> tuple[int, int]:
    """`(polynomial, seed)` в 32-битной GPU-совместимой форме (выровнено по старшим битам).

    Тот же формат, что принимает `FMCorrelatorROCm.set_params(polynomial=..., seed=...)`
    (H2 спеки) — передав эту пару в коррелятор и вызвав `generate_msequence(seed)`,
    получаем идентичный (бит-в-бит) результат, что и наш `m_sequence(degree, seed,
    poly_taps)`.
    """
    if not (0 < degree < _REGISTER_BITS):
        raise ValueError(f"degree должен быть в (0, {_REGISTER_BITS}), получено {degree}")
    mask = _taps_mask(degree, poly_taps)
    if not (0 < seed < (1 << degree)):
        raise ValueError(f"seed должен быть в (0, 2**degree={1 << degree}), получено {seed}")
    align = _REGISTER_BITS - degree
    polynomial = (mask << align) & _MASK32
    seed32 = (seed << align) & _MASK32
    return polynomial, seed32


def _run_lfsr(polynomial: int, seed32: int, length: int) -> np.ndarray:
    """Прогнать Galois LFSR (32-битный регистр) `length` тактов → ±1 float32[length]."""
    seq = np.empty(length, dtype=np.float32)
    lfsr = seed32
    for i in range(length):
        bit = (lfsr >> 31) & 1
        seq[i] = 1.0 if bit else -1.0
        if bit:  # noqa: SIM108 — тернарник с побитовыми операторами читается хуже
            lfsr = ((lfsr << 1) & _MASK32) ^ polynomial
        else:
            lfsr = (lfsr << 1) & _MASK32
    return seq


def m_sequence(
    degree: int = DEFAULT_DEGREE,
    seed: int = DEFAULT_SEED,
    poly_taps: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Максимальной длины (2^degree − 1) M-послед. ±1 (float32) через LFSR (Galois).

    `poly_taps` — отводы примитивного полинома (exponents, без x^degree и x^0);
    `None` → берётся из встроенной таблицы `_PRIMITIVE_TAPS` для `degree`.
    Валидируется тестом автокорреляции (thumbtack) — неверный полином → плохой
    пик → тест упадёт.
    """
    polynomial, seed32 = gpu_lfsr_params(degree, seed, poly_taps)
    length = (1 << degree) - 1
    return _run_lfsr(polynomial, seed32, length)


def m_sequence_pow2(
    degree: int = DEFAULT_DEGREE,
    seed: int = DEFAULT_SEED,
    poly_taps: tuple[int, ...] | None = None,
) -> np.ndarray:
    """`m_sequence(...)` продолженная ещё на один такт → длина `2**degree` (power-of-2).

    H3 (спека/таск P4): GPU-коррелятор требует `fft_size` степенью двойки, а
    период M-послед. `L=2^degree-1` таковым никогда не бывает. Здесь НЕ
    zero-pad (это ломает циклическую структуру для FFT-корреляции) — просто
    честно продолжаем ту же LFSR-траекторию ещё на один шаг: т.к. после `L`
    тактов состояние LFSR возвращается ровно в `seed` (полный период),
    `L`-й (следующий) выход **гарантированно равен** первому (`seq[0]`) —
    результат `2**degree`-периодичен по построению, циклические сдвиги внутри
    этого окна корректны для GPU-корреляции (см. отчёт P4/H3).
    """
    polynomial, seed32 = gpu_lfsr_params(degree, seed, poly_taps)
    length = 1 << degree
    return _run_lfsr(polynomial, seed32, length)
