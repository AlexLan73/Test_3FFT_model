# 🧩 TASK — Генераторы сигналов · P1 (NumpyBackend + Waveform: CW/ЛЧМ/АМ → SignalField)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** новый код + вендоринг формул DSP-GPU.
> **Спека:** [`specs/signal_generators_2026-07-13.md`](../specs/signal_generators_2026-07-13.md) (§4,§5).
> **Зависит от:** P0 (SignalField, WaveformSpec, TimeWindow, конфиг). **Статус:** ⬜ НЕ НАЧАТО.
>
> 🚨 🚫 pytest (только `TestRunner`) · 🚫 `.claude/worktrees/**` · существующее НЕ ломать.

---

## 🎯 Цель P1

Эталонная генерация на **чистом numpy** (CPU/Windows): базовый бэкенд + волны **CW, ЛЧМ, АМ** →
каждая отдаёт `SignalField` с сырым временем `[nx, ny, n_samples]` на n×n апертуре. **Без GPU** (P2),
**без ФМн/ЧМ/помех** (P4/P5), **без тактов** (P6).

## 📦 Что создать

### 1. `core/generators/backends/base.py` — `GenBackend(Protocol)` (Strategy/DIP, §4.3)
```python
class GenBackend(Protocol):
    def exp_phase(self, phase: np.ndarray) -> np.ndarray: ...          # exp(1j·phase)
    def apply_window(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray: ...  # x·mask, вне=0
    def add_noise(self, x: np.ndarray, power: float,
                  rng: np.random.Generator) -> np.ndarray: ...         # + комплексный AWGN
```

### 2. `core/generators/backends/numpy_backend.py` — `NumpyBackend(GenBackend)`
- Чистый numpy. `add_noise`: `scale=√(power/2)`, `x + scale·(randn+1j·randn)` (как наш
  `ThermalNoise`, `core/generators/sources.py:65`). **R5-математика:** дисперсия комплексного шума
  `σ²=power` (I+Q суммарно) — держать согласованно с `snr_db`-калибровкой (см. ниже).

### 3. `core/generators/waveforms/reference.py` — ВЕНДОРЕННЫЕ формулы (копия + атрибуция)
Шапка файла: `# vendored from DSP-GPU/DSP/Python/signal_generators/factories.py (2026-07-14)`.
Портировать **как есть** (numpy-эталон), не переизобретать:
- `cw(fs, n, f0, amp, phase) -> complex64[n]` ← `cw_numpy` (factories.py:62).
- `lfm(fs, n, f_start, f_end, amp) -> complex64[n]` ← `lfm_numpy` (factories.py:69).
- `getx(fs, n, f0, amp, phase, fdev, tau, ...) -> complex64[n]` ← `getX_numpy` (factories.py:79),
  **с оконной маской** `in_window` (это опорная формула размещения из P0 `TimeWindow`).

### 4. Волны (Strategy) — `am.py`, `lfm.py`, `cw.py` в `core/generators/waveforms/`
Каждая: `class XxxWaveform(Waveform)` c `.render(backend, spec, rng) -> SignalField`.
Пайплайн `render` (единый, §4.0 — данные формируются одинаково):
1. базовый 1D-сигнал во времени по формуле (реюз `reference.*`);
2. **окно** `spec.window.mask(...)` через `backend.apply_window`;
3. **раскладка на n×n апертуру**: `grid.steering(kx, ky)` (наш `core/generators/grid.py`) →
   `field[nx,ny,n] = steer[nx,ny,None] · sig[None,None,n]` (как `PointTarget.contribute`,
   `sources.py:54`). ⚠️ **геометрию задаёт SceneModeler** (§4.6, R3) — сюда угол приходит в spec/params,
   волна сама геометрию не решает;
4. **шум** по `snr_db` (если задан): амплитуда сигнала `A=√(σ²·10^(snr_db/10))` для **комплексного**
   baseband (R5 — БЕЗ множителя 2, т.к. IQ), затем `backend.add_noise`;
5. упаковка в `SignalField(data, modulation=..., axes=(ANTENNA_X,ANTENNA_Y,FAST_TIME), fs, carrier, meta)`.

Формулы:
- **CW:** `reference.cw` (опора).
- **ЛЧМ:** `reference.lfm`/`getx` (`f_end=f_start+fdev`).
- **АМ (новая, §5):** `a(t)=(1+m·cos(2π·f_m·t))·exp(j·2π·f0·t)`; `m`,`f_m` — в `spec.meta`.

### 5. Фабрика — реюз `EmitterFactory` (§4.2, «не плодить фабрики»)
Зарегистрировать новые волны в **существующем** `core/generators/factory.py` реестре (или тонкий
`WaveformFactory`, если контракт `EmitterFactory` не подходит — обосновать на ревью, не молча).

## ♻️ Реюз (точные пути)
- `cw_numpy`/`lfm_numpy`/`getX_numpy` — `DSP-GPU/DSP/Python/signal_generators/factories.py`.
- Стиринг/раскладка — `core/generators/grid.py` (`ArrayGrid.steering`), пример `sources.py:54` (`PointTarget`).
- Шум — `core/generators/sources.py:65` (`ThermalNoise`).

## ✅ Definition of Done
- Тесты (`tests/test_generators.py`, `TestRunner`):
  - **CW:** пик спектра `np.fft.fft` на `f0` (реюз идеи `_check_peak_frequency`, `signal_base.py:115`).
  - **ЛЧМ:** мгновенная частота линейна (спектрограмма/производная фазы); совпадение с `reference.lfm`.
  - **АМ:** в спектре несущая + 2 боковые на `f0±f_m`.
  - **Окно:** `short`/`partial` → энергия вне окна ≈ 0; полезный сигнал в нужном месте.
  - **SNR:** при `snr_db` заданном — измеренный SNR ≈ заданному ±1 дБ. ⚠️ `SnrEstimator.estimate(signal)`
    работает по **1D**-ряду → мерить на срезе одной антенны `field.data[i, j, :]`, не на всём кубе.
  - **Форма/тип:** `SignalField.data.shape==(16,16,n)`, dtype complex64, `axes` корректны.
  - **Детерминизм (R6):** один seed → идентичный результат (побитово).
- `ruff`/`mypy` чисто. Набор в `tests/all_test.py`.

## 🖼️ Визуал-подтверждение (обязательно, §9-конвенция)
Каталог: **`graphics/signal_generators/p1_numpy_cw_lfm_am/`** (`FigureWriter`, стиль `example_form_signal.py`).
Демо в `demo_generators.py` (Composition Root: `NumpyBackend`, генерация CW/ЛЧМ/АМ):
- `cw_time_spectrum.png` — Re/|·| времени + спектр (`np.fft`): один пик на `f0` (подтверждает CW).
- `lfm_spectrogram.png` — **спектрограмма ЛЧМ** (`scipy.signal.spectrogram`): наклонная линия = линейный чирп.
- `am_spectrum.png` — спектр АМ: несущая `f0` + 2 боковые `f0±f_m` (подтверждает модуляцию).
- `window_placement.png` — |сигнал(t)| для `full`/`partial`/`short`: энергия только внутри окна (§0.3).
- `snr_check.png` — измеренный (через `core/snr`) vs заданный `snr_db` для набора значений (диагональ ±1 дБ).

## ⚠️ Подводные камни
- **R5-математика:** IQ baseband → `A=√(σ²·10^(SNR/10))` без ×2; сверить измеренный SNR тестом.
- **R6:** rng — только переданный `Generator`.
- **R3:** волна геометрию НЕ знает — угол/задержка приходят снаружи (заглушка «один источник по нормали»
  для P1 ок; полноценный `SceneModeler` — позже).
- Не мутировать входные массивы (чистота, `sources.py` стиль).

## 🚫 Вне P1
GPU (`HipBackend`), дечирп/FFT/куб, ФМн-код, ЧМ, помехи, движение/такты.
