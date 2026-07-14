# 🧩 TASK — Генераторы сигналов · P4 (ФМн-код 2ⁿ + коррелятор + ЧМ-помеха)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** новый код + реюз готового GPU-коррелятора.
> **Спека:** [`specs/signal_generators_2026-07-13.md`](../specs/signal_generators_2026-07-13.md) (§5, §6.2, §3.1).
> **Зависит от:** P0+P1+P2 (`SignalField`, `Waveform`, `NumpyBackend`, `_pipeline`, `HipBackend`, gpu_libs).
> **Статус:** ⬜ НЕ НАЧАТО · 🔍 **ревью Кодо 2026-07-14** (сверено с кодом DSP-GPU): внесены H1 (коррелятор
> REAL-код, не комплекс — гл. фикс), H2 (сверка LFSR ↔ `generate_msequence`), H3 (период `L` vs `fft_size`).
>
> ℹ️ Старый **P3** (окно full/partial/short + шум по SNR) **уже сделан в P1** (`TimeWindow` + `render_pipeline`
> + `add_noise`) — отдельной фазой не нужен. Это P4.
>
> 🚨 🚫 pytest (только `TestRunner`+`AssertionGroup`+`SkipTest`) · 🚫 `.claude/worktrees/**` · существующее НЕ ломать.

---

## 🎯 Цель P4

1. **ФМн-зонд** (FM-m патента, гл.6): генерация M-последовательности (LFSR, **наша** — переносим из
   коррелятора, §6.2) + BPSK-модуляция на несущую → `SignalField` на n×n (как CW/ЛЧМ).
2. **Реюз готового GPU-коррелятора** `dsp_radar.FMCorrelatorROCm`: наш код → `prepare_reference_from_data`
   → `process` → пик корреляции `[S,K,n_kg]` (это представление «ФМн→задержка», §4.0).
3. **ЧМ-помеха** (аналоговая FM): сторонний источник/связной → `SignalField`.

## 📦 Что создать

### 1. `core/generators/waveforms/mseq.py` — LFSR M-последовательность (наша, numpy)
```python
def m_sequence(degree: int, seed: int = 1, poly_taps: tuple[int, ...] | None = None) -> np.ndarray:
    """Максимальной длины (2^degree − 1) M-послед. ±1 (float32) через LFSR (Фибоначчи/Галуа).

    poly_taps — отводы примитивного полинома; None → взять из встроенной таблицы для degree.
    Валидируется тестом автокорреляции (thumbtack) — неверный полином → плохой пик → тест упадёт.
    """
```
- Встроенная таблица **примитивных** полиномов для degree ~7..16 (стандартные, из таблиц LFSR).
- Дефолтный `degree` подобрать так, чтобы `2^degree−1` укладывалось в `n_samples` baseline (8192):
  **degree=13 → 8191 чипов** (ровно под 8192). Длину/полином/seed можно менять (Q10: от такта к такту).
- Выход ±1 float32 (совместимо с коррелятором — он на R2C FFT, вход real float32).

### 2. `core/generators/waveforms/phase_code.py` — `PhaseCodeWaveform(Waveform)` (ФМн)
- Шаг 1 (несущая+код): BPSK **напрямую через код ±1** (проще, чем конверсия в `{0,1}`):
  `s(t) = amplitude · code(t) · exp(j·(2π·f0·t + phase))` (эквивалент `exp(jπ·c)`, но без перевода),
  где `code(t)` — чип ±1 по времени (L чипов растянуты на `n_samples`: `chip = floor(k·L/n_samples)`).
  Параметры (`degree`,`seed`,`poly`) — из `spec.meta` (G10), дефолты из `mseq`.
- Шаги 2-5: **реюз `_pipeline.render_pipeline(backend, spec, rng, s, Modulation.PHASE_CODE)`** (окно/n×n/шум).
- 🔴 **H1:** `render` отдаёт **комплексный passband** `SignalField` (для спектра/датасета). Готовый GPU-коррелятор
  работает **НЕ с ним**, а с **реальным ±1 кодом baseband** (см. §5). Сам код (float32 ±1, до несущей) сохранить
  в `field.meta["code"]` (или вернуть отдельно) — именно его подаём коррелятору, а не `SignalField.data`.

### 3. `core/generators/waveforms/fm.py` — `FmInterferenceWaveform(Waveform)` (ЧМ-помеха, §5)
- `s(t) = amplitude · exp(j·(2π·f0·t + β·sin(2π·f_m·t) + phase))` (интеграл косинуса = синус;
  `β` — индекс ЧМ, `f_m` — частота модуляции; в `spec.meta`). `Modulation.FM_INTERFERENCE`.
- Реюз `render_pipeline`. Это модель радиолюбителя/связного (не наш зонд).

### 4. Регистрация в `WaveformFactory` (реюз, §4.2) — `PHASE_CODE`/`FM_INTERFERENCE` → волны.

### 5. Реюз коррелятора (GPU) — прямой вызов в тесте (Converter — позже, см. H1)
🔴 **H1 (сверено с кодом `py_fm_correlator_rocm.hpp:186,192` + тест `t_fm_correlator_rocm.py`):**
`process(input_signals)` и `prepare_reference_from_data(ref)` принимают **REAL `float32 [S, N]`**
(пайплайн R2C-FFT). Коррелятор — **кодовый** (baseband ±1), НЕ комплексный passband. Поэтому:
- вход/референс — **реальный ±1 код** (`m_sequence(...).astype(np.float32)`), а НЕ `SignalField.data`;
- 🐞 `signals.astype(np.float32)` на **комплексном** массиве **молча роняет мнимую часть** (ComplexWarning) —
  так делать нельзя; код формируем real float32 с самого начала.
```python
corr = dsp_radar.FMCorrelatorROCm(ctx)
corr.set_params(fft_size=N, num_shifts=K, num_signals=S, num_output_points=n_kg)  # poly/seed игнор при ref-from-data
code = m_sequence(degree=13).astype(np.float32)                 # НАШ real ±1 код (не комплекс!)
corr.prepare_reference_from_data(code)                          # референс = наш код
signals = np.stack([np.roll(code, -d) for d in shifts]).astype(np.float32)  # [S, N] real
peaks = corr.process(signals)                                  # [S, K, n_kg]
```
- Полноценный `RawToCorrelation` (комплексный `SignalField` → real baseband ±1: домножить на `conj(carrier)`
  → despread → `sign`) **нетривиален** → в P4 интероп делаем **на уровне кода** (real m-seq); конвертор — позже.
- Образец: `DSP-GPU/DSP/Python/radar/t_fm_correlator_rocm.py:48-52,95-103` (кормит `ref`/`np.roll(ref)`, real).
- Под `python3.13` (cp313 .so, как P2). Под `.venv`/без ROCm → `SkipTest`.

## ♻️ Реюз (точные пути — читать перед кодом)
- Коррелятор API + образец: `DSP-GPU/DSP/Python/radar/t_fm_correlator_rocm.py`, py-обёртка
  `DSP-GPU/radar/python/py_fm_correlator_rocm.hpp` (сигнатуры `set_params/generate_msequence/
  prepare_reference_from_data/process`).
- Пайплайн/бэкенд: `core/generators/waveforms/_pipeline.py`, `backends/{numpy_backend,hip_backend}.py`.
- gpu_libs loader (P2): `core/gpu_libs/loader.py`.

## ✅ Definition of Done
- Тесты (`tests/test_generators.py`, `TestRunner`):
  - **Автокорреляция (главный, numpy):** циклическая автокорреляция `m_sequence(13)` = **thumbtack** —
    пик `L` при нулевом сдвиге, `≈ −1` на всех прочих (свойство M-послед.). Ловит неверный полином.
  - **LFSR-сверка (H2, сильный тест, GPU/SkipTest):** наш `m_sequence(poly,seed)` **==**
    `corr.generate_msequence(seed)` для ТОГО ЖЕ полинома/seed (побитово, с точностью до known-конвенции
    отводов) — доказывает совпадение нашего LFSR с эталонным (и что таблица примитивных полиномов верна).
  - **ФМн-спектр:** BPSK расширяет спектр (широкополосный), не узкий тон.
  - **ЧМ:** спектр ЧМ — несущая + боковые Бесселя (или спектрограмма — постоянная частота с девиацией).
  - **Форма/тип/детерминизм:** `SignalField` (16,16,n) complex64, один seed → идентично (R6).
  - **Коррелятор-интероп (GPU, SkipTest):** наш **real ±1** `m_sequence` как ref, циклически сдвинутый на `d`
    (real float32 `[S,N]`) как вход → пик `process()` на позиции `d`. НЕ комплексный `SignalField`. Реально на RX 9070.
- `.venv/bin/python tests/all_test.py` — зелено (GPU-тест скип). `/usr/bin/python3.13 tests/all_test.py` —
  коррелятор реально прогнан.
- `ruff check core/generators tests` + `mypy core/generators` — чисто.

## 🖼️ Визуал-подтверждение (обязательно, §9-конвенция)
Каталог `graphics/signal_generators/p4_phase_code_fm/` (`FigureWriter`), демо в `demo_generators.py`:
- `mseq_autocorr.png` — thumbtack автокорреляции (пик L, фон ≈−1) — доказывает корректность кода.
- `phase_code_spectrum.png` — код (±1 по времени) + широкополосный спектр ФМн.
- `fm_interference.png` — спектрограмма/спектр ЧМ (несущая + девиация).
- `correlator_peak.png` — выход `FMCorrelatorROCm`: острый пик на позиции сдвига (GPU; SkipTest без ROCm).

## ⚠️ Подводные камни
- **Длина кода vs n_samples:** `2^degree−1` должно укладываться в `n_samples` (degree=13→8191 под 8192).
  Не брать дефолтный полином коррелятора `0x00400007` (degree 22 → 4.2M чипов, не влезет).
- 🔴 **H1 — коррелятор REAL, не комплекс:** референс/вход — real float32 ±1 код (`m_sequence`), НЕ `SignalField.data`.
  `.astype(np.float32)` на **комплексе** роняет Im (ComplexWarning) → формировать код real с самого начала.
- **H3 — период `L` vs `fft_size`:** thumbtack точен на периоде `L=2^deg−1` (8191 — НЕ степень 2). numpy-автокорр
  тестим на периоде `L`; GPU-коррелятор FFT'ит по `fft_size` (pow2 ≥ L, напр. 8192) с zero-pad — позиции пика
  трактуем по схеме `t_fm_correlator_rocm.py`, не приравнивать вслепую к циклической автокорр длины `L`.
- **R6:** rng — переданный `Generator` (шум/выбор кода).
- **cp313 для GPU-коррелятора** (как P2): `.so` под 3.13; под .venv → SkipTest.
- НЕ ломать P0/P1/P2; ФМн/ЧМ — новые файлы, реюз `render_pipeline`, не дублировать пайплайн.

## 🚫 Вне P4
Помехи-зоопарк (SMSP/DRFM-repeater/BarrageRF/промышленные — это P5), движение/такты (P6),
порт в C++ (P7). Здесь — только ФМн-зонд + коррелятор-интероп + ЧМ-помеха.
