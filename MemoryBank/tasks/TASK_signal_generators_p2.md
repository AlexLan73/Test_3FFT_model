# 🧩 TASK — Генераторы сигналов · P2 (gpu_libs loader + GPU-smoke + HipBackend + сверка)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** обвязка готовых GPU-моделей DSP-GPU.
> **Спека:** [`specs/signal_generators_2026-07-13.md`](../specs/signal_generators_2026-07-13.md) (§2.2,§2.2.1,§4.3,§8).
> **Зависит от:** P0+P1 (`SignalField`, `NumpyBackend`, волны, тесты). **Статус:** ✅ ПРИНЯТО
> (ревью Кодо 2026-07-14). GPU на RX 9070: python3.13 → 26 ok/0 fail; .venv → GPU скип чисто.
> CW/ЛЧМ N=4096 raw=5.11e-4 (<1e-3); модуль N=8192=1.19e-7 (формула точна). Кодо добавил тест
> модуля N=8192 (закрыл дыру находки). `.so` в gitignore. Вендоринг с атрибуцией.

---

## ✅ РЕЗУЛЬТАТ ШАГОВ 1-3 (Sonnet, 2026-07-14)

**Файлы:**
- ШАГ 1: `core/gpu_libs/{__init__.py,loader.py,configGPU.json,sync_gpu_libs.sh}` (+5 `.so`,
  синкнуты `sync_gpu_libs.sh`, НЕ в git — `*.so` уже покрыт корневым `.gitignore`).
- ШАГ 2: `common/gpu_context.py` (вендорено+адаптировано: свой `core.gpu_libs.loader` вместо
  DSP-GPU `GPULoader`), `common/gpu_configs.py` (подмножество `configs.py` — только парсинг
  `configGPU.json`), `common/result.py` + `common/validators/{__init__,base,composite,factory,
  numeric,signal}.py` (вендорено 1:1 + косметика под ruff: `List/Optional`→`list/X|None`,
  сортировка импортов, снят 1 неисп. импорт — логика не менялась), `core/generators/backends/
  hip_backend.py` (`HipBackend`, **не** реэкспортится из `backends/__init__.py` — опциональный
  GPU-модуль, явный импорт `from core.generators.backends.hip_backend import HipBackend`,
  чтобы не тянуть импорт `waveforms` в базовый `backends`-пакет при каждом `import
  core.generators.backends`).
- ШАГ 3: 4 новых теста в `tests/test_generators.py` (`test_hip_cw_matches_numpy`,
  `test_hip_lfm_matches_numpy`, `test_hip_lfm_matches_getx_reference` — G11-регресс,
  `test_hip_backend_rejects_unsupported_modulation`), `demo_generators.py` (+`_demo_gpu_vs_numpy`,
  ветка по доступности `.so`/ROCm).

**Архитектурное решение (не в таске, понадобилось при реализации):** `Waveform.render()`
(`CwWaveform`/`LfmWaveform`) всегда считает несущую через `reference.cw_numpy`/`getX_numpy`
(numpy), т.е. `backend`-параметр в них влияет только на окно/шум, не на синтез — значит просто
реализовать `HipBackend(GenBackend)` с теми же 3 примитивами было бы недостаточно (GPU не
участвовал бы в генерации несущей). Добавлен **`HipBackend.render(modulation, spec, rng)`** —
собственный метод (не входит в `GenBackend`-протокол, но **тот же результат** `SignalField`):
шаг 1 (несущая) считает `FormSignalGeneratorROCm.generate()` на 1 антенне вместо
`cw_numpy`/`getX_numpy`, шаги 2-5 (окно/раскладка n×n/шум) — **переиспользуют**
`waveforms._pipeline.render_pipeline(self, spec, rng, raw, modulation)` (тот же код, что и
`NumpyBackend`-путь — DRY, ноль дублирования логики). Проверено: циклический импорт
`backends↔waveforms` не возникает, т.к. `hip_backend.py` — не eager-импорт `backends/__init__.py`
(см. комментарий в файле).

**Реальные числа (`/usr/bin/python3.13`, RX 9070, `DataValidator(metric="max_rel")`, N=4096 —
как у принятых P1-тестов `test_lfm_matches_reference`/`test_cw_peak_frequency`):**

| Тест | max_rel | порог | результат |
|------|---------|-------|-----------|
| CW: HipBackend ↔ NumpyBackend | **5.11e-4** | 1e-3 | ✅ PASS |
| ЛЧМ (fdev=1e6): HipBackend ↔ NumpyBackend | **5.11e-4** | 1e-3 | ✅ PASS |
| ЛЧМ: HipBackend ↔ `reference.getX_numpy(norm=1)` (G11-регресс) | **5.11e-4** | 1e-3 | ✅ PASS |

**🐞 Новая находка (не блокер, честно фиксирую для ревью):** на **N=8192-baseline** (§5.1)
`max_rel` **превышает** согласованный порог 1e-3, а не только старый 1e-4:
CW N=8192 → **1.023e-3** ❌, ЛЧМ N=8192 fdev=6e6 (§5.1 полоса) → **1.789e-3** ❌ (растёт с N и с
`fdev`, тот же float32-фазовый эффект, что в эскалации G11 находка #1). Таблица роста (CW,
`amplitude=1`, без окна/шума):
`N=512→6.4e-5, 1024→1.3e-4, 2048→2.6e-4, 4096→5.1e-4, 8192→1.02e-3`.
**Решение (принято здесь, не эскалирую повторно):** тесты `test_hip_*_matches_*` держат
`N=4096` (запас x2 от порога, тот же N что уже приняты в P1-precision-тестах — не новая
договорённость, а следование существующему прецеденту). Демо (`gpu_vs_numpy_*.png`) тоже
на N=4096. Baseline N=8192 (полный конфиг `default_scenario`/§5.1) **не** проверяется raw-complex
сравнением на этом пороге — при необходимости трекать отдельно (шум/окно/интеграция по кубу
могут сгладить фазовый дрейф; не проверено в P2). Если Alex/Кодо хотят строгую проверку именно
на N=8192 — нужно либо увеличить порог для этого N, либо перейти на посегментное сравнение (не
делал — вне объёма P2, не додумываю).

**Прогон тестов:**
- `.venv/bin/python tests/all_test.py` → exit 0, `GeneratorsTests: 20 ok · 0 fail · 5 skip`
  (4 GPU-теста + `test_yaml_config_source` — pyyaml не стоит в `.venv`), остальные 48
  (Smoke/Graphics/Nuller/Cfar/Snr) не сломаны.
- `/usr/bin/python3.13 tests/all_test.py` → exit 0, `GeneratorsTests: 25 ok · 0 fail · 0 skip`
  (все 4 GPU-теста реально прогнаны на RX 9070, числа выше).

**git:** `.so` НЕ коммитятся (`*.so` уже в корневом `.gitignore` с самого начала — отдельная
запись под `core/gpu_libs/` не понадобилась). `graphics/` целиком в `.gitignore` (уже было) —
`p2_gpu_vs_numpy/*.png` туда же, генерируются `demo_generators.py`, не коммитятся.

**ruff/mypy:** `ruff check` — чисто на всех новых/вендоренных файлах (после автофикса
`List/Optional`→`list/X|None`+сортировка импортов + переименование `GpuLibsUnavailable`→
`GpuLibsUnavailableError`, N818). `mypy` — чисто, кроме одного `# type: ignore[call-arg]` в
вендоренном `common/validators/factory.py` (мypy не может проверить `__init__` через
`dict[str, type[IValidator]]`+динамическую диспетчеризацию — ограничение инструмента, не баг).

**Визуал:** `graphics/signal_generators/p2_gpu_vs_numpy/{gpu_vs_numpy_overlay,
gpu_vs_numpy_error}.png` — сгенерены `/usr/bin/python3.13 demo_generators.py`, оверлей
Re(t) GPU/NumPy визуально совпадает (CW и ЛЧМ), карта ошибки растёт со временем (float32-фаза),
остаётся ниже порога 1e-3·max|ref| на всём окне N=4096.

---

## ✅ РЕШЕНИЯ ПО ЭСКАЛАЦИИ (Alex 2026-07-14) — эскалация закрыта

1. **Порог сверки GPU↔numpy = `max_rel < 1e-3`** для **сырого комплекса** (не 1e-4; float32-дрейф фазы
   GPU-ядра растёт с N — как сам DSP-GPU `demo_gpu_vs_numpy`). Спека §8 обновлена.
2. **Каноническая ЛЧМ = `getX` (центрированный чирп).** P1 `LfmWaveform` **уже пропатчен** Кодо на
   `reference.getX_numpy` (совпадает с боевым GPU `FormSignalGeneratorROCm`). Тесты/визуал перегенерены,
   48 зелены. → `HipBackend(LFM)` теперь сойдётся с `NumpyBackend(LFM)` в пределах 1e-3.
3. **`HipBackend` вызывает `set_params(..., norm=1.0)`** (GPU-дефолт `norm=1/√2` — иначе амплитуда ×0.7071).

**Реальная сигнатура (из smoke, подтверждена):**
`ctx=dsp_core.ROCmGPUContext(0)`; `gen=dsp_signal_generators.FormSignalGeneratorROCm(ctx)`;
`gen.set_params(antennas, points, fs, f0, amplitude, phase, fdev, norm, noise_amplitude, noise_seed, tau_base, tau_step, tau_min, tau_max, tau_seed)`; `gen.generate() -> complex64[antennas, points]`.

---
>
> 🚨 🚫 pytest (только `TestRunner`) · 🚫 `.claude/worktrees/**`.
> 🐧 **Только Linux+ROCm.** Всё, что GPU, — под `SkipTest`, если нет `.so`/ROCm (Windows/CI).

---

## 🔴 РЕЗУЛЬТАТ ШАГА 0 (smoke, 2026-07-14) — СТОП, эскалация Alex/Кодо

**Окружение:** `/usr/bin/python3.13`, `DSP-GPU/DSP/Python/libs/dsp_*.cpython-313-x86_64-linux-gnu.so`,
GPU обнаружен и реально использован: `ROCmGPUContext(0)` → `AMD Radeon RX 9070` (gfx1201, 16304 MB).

### Сигнатура (реальная, сверено `help()`)
```python
import dsp_core as core
import dsp_signal_generators as sg

ctx = core.ROCmGPUContext(0)                 # НЕ gpuworklib.GPUContext — другое имя (подтверждено спекой)
gen = sg.FormSignalGeneratorROCm(ctx)
gen.set_params(antennas=1, points=8192, fs=12e6, f0=2e6,
               amplitude=1.0, phase=0.0, fdev=0.0,
               norm=0.7071067811865476,        # ⚠️ ДЕФОЛТ norm=1/√2, НЕ 1.0!
               noise_amplitude=0.0, noise_seed=0,
               tau_base=0.0, tau_step=0.0, tau_min=0.0, tau_max=0.0, tau_seed=12345)
data = gen.generate()                          # complex64[antennas, points] — форма ОК, dtype ОК
```
Чтобы сверить с нашим `reference.cw_numpy`/`lfm_numpy` (без множителя `norm`), нужно явно
передавать `norm=1.0` (иначе амплитуда GPU-сигнала домножается на 1/√2 ≈ 0.7071, что тоже
формально "не то же самое", но это ожидаемо и решается передачей параметра — не блокер).

### 🐞 G11-находка #1 — CW: `max_rel` растёт с `N`, на baseline (`N=8192`) выше порога 1e-4
`gen.generate()[0]` (`norm=1.0`) против `reference.cw_numpy(fs=12e6, f0=2e6, ...)`:

| N     | max_rel = max\|a−r\|/max\|r\| |
|-------|-------------------------------|
| 512   | 6.39e-05                       |
| 1024  | 1.28e-04                       |
| 2048  | 2.56e-04                       |
| 4096  | 5.11e-04                       |
| 8192 (baseline §5.1) | **1.02e-03** ❌ (> 1e-4) |

Погрешность растёт **линейно с N** (∝ фаза 2π·f0·t на конце окна) — классический эффект
float32-точности GPU-ядра (фаза/`t` считаются в float32 на GPU, а не в float64 как в NumPy):
абсолютная погрешность фазы ≈ (2π·f0·T)·2⁻²⁴ ≈ наблюдаемой ошибке. Это НЕ баг формулы/нормировки —
это предел точности float32-кернела. **Важно:** сам DSP-GPU в своих тестах ПРИНИМАЕТ этот эффект —
`t_form_signal_rocm.py` сравнивает только `|gpu|` vs `|ref|` (`atol=1e-4`, модуль не дрейфует),
а `example_form_signal.py::demo_gpu_vs_numpy` явно использует порог **`1e-3`**, не `1e-4`:
`"GPU matches NumPy: YES if err < 1e-3"`. Порог `1e-4`, заданный в этом таске (и спеке §8), для
сырого комплексного сравнения на `N=8192` **физически недостижим** на float32 GPU-ядре.

### 🐞 G11-находка #2 (более серьёзная) — ЛЧМ: формула GPU ≠ формула нашего P1 `LfmWaveform`
GPU `FormSignalGeneratorROCm` при `fdev≠0` реализует **`getX`-формулу** (квадратичная фаза
**центрирована на середине окна** `ti/2`):
```
X = a·norm·exp(j·(2π·f0·t + π·fdev/ti·(t−ti/2)² + phi))
```
Это **байт-в-байт формула нашего же `waveforms/reference.py::getX_numpy`** (max_rel=5.1e-4 на
N=4096 — та же float32-погрешность, что и у CW, формула СОВПАДАЕТ).

НО наш P1 `LfmWaveform` (`waveforms/lfm.py`) вызывает **другую** функцию — `reference.lfm_numpy`
(линейный чирп `f0→f0+fdev`, фаза отсчитывается от `t=0`, БЕЗ центрирования):
```
phase = 2π·(f_start·t + 0.5·chirp_rate·t²),  f_start=carrier_hz, f_end=carrier_hz+fdev_hz
```
Сверка `gen.generate()[0]` (GPU, fdev=1e6) vs `reference.lfm_numpy(...)`: **`max_rel = 2.0`**
(практически полная некогерентность, НЕ погрешность округления). Т.е. `HipBackend` (боевой,
через `FormSignalGeneratorROCm`) и уже принятый `NumpyBackend`+`LfmWaveform` (P1) **синтезируют
физически разные ЛЧМ-сигналы** (разный частотный диапазон: GPU фактически метёт
`[f0−fdev/2, f0+fdev/2]` вокруг несущей, наш `LfmWaveform` метёт `[f0, f0+fdev]` от несущей вверх).

Важно: обе formulы (`lfm_numpy` и `getX_numpy`) существуют **в самом DSP-GPU**
(`DSP/Python/signal_generators/factories.py:69,79`) как ДВЕ РАЗНЫЕ, не эквивалентные друг другу
функции — это не ошибка вендоринга с нашей стороны, а факт исходника. Наш P1 (уже принятый на
ревью) выбрал `lfm_numpy` для `LfmWaveform`, а боевой `.so` реализует `getX_numpy`.

### ⛔ Почему это блокер Шага 0 (не додумываю — эскалирую)
1. Порог `max_rel<1e-4` из спеки/таска для сырого комплексного сравнения на CW при N=8192
   недостижим на float32 GPU (физика float32, не код) — DSP-GPU сам использует `1e-3` для той же
   проверки. Нужно решение: либо (a) принять `1e-3` как рабочий порог для raw-complex сравнения
   (как в самом DSP-GPU), либо (b) сравнивать через `RelativeValidator`/на модуле+фазе отдельно,
   либо (c) держать `1e-4`, но только для меньших `N` — Alex/Кодо решают, я не выбираю сам.
2. ЛЧМ: `HipBackend` через `FormSignalGeneratorROCm` физически НЕ совпадает с принятым P1
   `LfmWaveform`. Шаг 3 (`max_rel(HipBackend, NumpyBackend)<1e-4` на ЛЧМ) **гарантированно упадёт**
   при обоих «корректных» бэкендах — это ровно риск G11, который таск просил ловить здесь.
   Нужно решение архитектуры: либо (a) переписать `LfmWaveform` на `getX_numpy`-формулу (центр.
   чирп) — тогда P1-тесты `test_lfm_matches_reference`/`test_lfm_instantaneous_freq_linear`
   (уже приняты!) тоже надо менять, либо (b) `HipBackend` вызывает `.so` с параметрами,
   компенсирующими центрирование (сдвиг `f0`→`f0+fdev/2` и учёт `tau`?) — нужно проверить,
   даёт ли это тождественность, либо (c) взять `DelayedFormSignalGeneratorROCm`/иной GPU-класс
   с другой формулой (не проверено в этом smoke). Решение — за Alex/Кодо.

**Ничего из ШАГ 1-3 (gpu_libs/, HipBackend, вендоринг GPUContextManager/DataValidator, демо,
тесты) НЕ реализовано** — по прямому указанию таска "СТОП, не додумывать обвязку" при расхождении
на смоуке. Полный код проверки — см. отчёт сессии/это сообщение (voспроизводимо: см. команды ниже).

**Как воспроизвести:**
```bash
cd /home/alex/DSP-GPU/DSP/Python
/usr/bin/python3.13 -c "
import sys; sys.path.insert(0, 'libs')
import numpy as np, dsp_core as core, dsp_signal_generators as sg
ctx = core.ROCmGPUContext(0)
gen = sg.FormSignalGeneratorROCm(ctx)
gen.set_params(fs=12e6, f0=2e6, antennas=1, points=8192, norm=1.0)
data = gen.generate()
t = np.arange(8192)/12e6
ref = np.exp(1j*2*np.pi*2e6*t).astype(np.complex64)
print('CW max_rel:', np.max(np.abs(data[0]-ref)))
"
```

---

## 🎯 Цель P2

Подключить **боевой GPU-бэкенд** (собранные модели DSP-GPU, cp313) и доказать эквивалентность
`HipBackend == NumpyBackend` на генерации. Здесь закрываем **блокеры ревью R1 (загрузка .so)** и
**R2 (реальный GPU-run, а не только import)**.

## 🔴 ШАГ 0 — сначала smoke-run (R2), потом обвязка. НЕ пропускать!

Перед любым кодом `HipBackend` — **проверить вживую** (спека §2.2, ⚠️-блок):
```python
# smoke: реальный вызов, не только import
gen = dsp_signal_generators.FormSignalGeneratorROCm(ctx)
gen.set_params(fs=12e6, f0=2e6, antennas=1, points=8192, amplitude=1.0, noise_amplitude=0.0)
data = gen.generate()      # ← сверить: форма, dtype, что не падает
```
- Сверить **точную сигнатуру** `set_params`/`generate` (в старом примере было имя
  `gpuworklib.FormSignalGenerator` — ДРУГОЕ; боевое — `dsp_signal_generators.FormSignalGeneratorROCm`).
- Результат smoke — записать в шапку этого TASK (реальная сигнатура + форма выхода).
- **Если API разошёлся со спекой — СТОП, эскалация Alex/Кодо**, не додумывать.
- 🐞 **G11 (ревью тасков):** smoke — не только форма/dtype, но и **числовая сверка формулы**:
  `max_rel(gen.generate()[0], reference.cw(fs,n,f0,...)) < 1e-4`. Эквивалентность `Hip↔Numpy` (шаг 3)
  держится ТОЛЬКО если `.so` реализует **ту же формулу**, что vendored `factories.py`. Если `.so` пересобран
  с другой нормировкой/фазой/окном — тест шага 3 упадёт при обоих «корректных» бэкендах. Ловим здесь.

## 📦 ШАГ 1 — `core/gpu_libs/` (R1, спека §2.2.1)
- Скопировать нужные `.so` (cp313) из `DSP-GPU/DSP/Python/libs/`:
  `dsp_core*.so, dsp_signal_generators*.so, dsp_heterodyne*.so, dsp_radar*.so, dsp_spectrum*.so`
  + `configGPU.json` (его ищет `GPUContextManager` рядом с `dsp_core.so`).
- `core/gpu_libs/loader.py` — тонкий: добавляет путь к libs в `sys.path`, импортирует `dsp_*`,
  отдаёт хэндлы. При отсутствии (`ImportError`/не Linux) — понятный `SkipTest`-сигнал.
- 🐞 **G12 (ревью тасков):** `.so` (cp313) — бинарники в несколько МБ. **Рекомендация:** НЕ коммитить их
  (замусорят историю) — `core/gpu_libs/*.so` в `.gitignore` + скрипт-синк `sync_gpu_libs.sh`
  (копирует из `DSP-GPU/DSP/Python/libs/` по надобности). `configGPU.json` + `loader.py` — **в git**.
  Финальное решение (git / git-lfs / sync-скрипт) — **за Alex** перед шагом 1.

## 📦 ШАГ 2 — `core/generators/backends/hip_backend.py` — `HipBackend(GenBackend)`
- Тот же контракт `GenBackend` (LSP), но считает на GPU через `FormSignalGeneratorROCm`.
- `GPUContextManager` — **вендорить** из DSP-GPU (см. §8, R8: тянет `gpu_context.py`+`gpu_loader.py`+
  `configs.py`; атрибуция на каждый файл). Один контекст на сессию (Singleton).
- **GPU-first (§4.3):** `HipBackend` — главный боевой; `NumpyBackend` — эталон.
- Возврат — тот же `SignalField` (формат идентичен P1).

## 📦 ШАГ 3 — сверка GPU↔NumPy (§8, эталонный тест)
- Вендорить `DataValidator(metric="max_rel")` — **полное замыкание** (R8): `validators/base.py`,
  `common/result.py` (`TestResult`/`ValidationResult`), при нужде `reporters.py`. Атрибуция на каждый.
  Кладём в `common/` нашего репо (рядом с `runner.py`), НЕ ломая `AssertionGroup`-механизм (правило 04).
- Тест: одинаковые параметры → `max_rel(HipBackend.data, NumpyBackend.data) < 1e-4`
  (как `demo_gpu_vs_numpy`, `example_form_signal.py:214`).

## ♻️ Реюз (точные пути)
- GPU-генератор: `dsp_signal_generators.FormSignalGeneratorROCm` (пример `DSP/Python/signal_generators/example_form_signal.py`).
- Контекст: `DSP-GPU/DSP/Python/common/gpu_context.py` (`GPUContextManager`), `gpu_loader.py`, `configs.py`.
- Валидатор: `DSP-GPU/DSP/Python/common/validators/` + `common/result.py`.
- ROCm-эталон-тест как образец: `DSP/Python/radar/t_fm_correlator_rocm.py`, `heterodyne/t_heterodyne_rocm.py`.

## ✅ Definition of Done
- **Smoke (R2)** пройден, реальная сигнатура зафиксирована в шапке TASK.
- `core/gpu_libs/loader.py` импортирует `dsp_*` на Linux; на Windows/без ROCm — чистый `SkipTest`.
- Тест `HipBackend ↔ NumpyBackend`: `max_rel < 1e-4` на CW/ЛЧМ (АМ — если DSP-GPU умеет; иначе только numpy).
- Под `SkipTest`, если нет GPU — тогда P2-тесты не падают, гоняется только numpy-ветка P1.
- `ruff`/`mypy` чисто (насколько применимо к обёрткам `.so`). Набор в `tests/all_test.py`.

## 🖼️ Визуал-подтверждение (обязательно, §9-конвенция)
Каталог: **`graphics/signal_generators/p2_gpu_vs_numpy/`** (`FigureWriter`).
Демо `demo_generators.py` дополнено веткой `HipBackend` (выбор бэкенда по платформе/флагу):
- `gpu_vs_numpy_overlay.png` — Re(t) GPU и NumPy на одних осях (визуально совпадают) для CW/ЛЧМ.
- `gpu_vs_numpy_error.png` — карта/график `|Hip−Numpy|` по времени (уровень < 1e-4, подтверждает LSP).
- ⚠️ на Windows/без ROCm графики P2 не строятся (`SkipTest`) — это норма.

## ⚠️ Подводные камни
- **R2:** НЕ строить обвязку до успешного smoke-run — API мог разойтись.
- **R1:** `GPUContextManager` требует `configGPU.json` рядом с `dsp_core.so` — иначе device=0/ошибка.
- **R8:** копий больше одной — тянется замыкание; каждую пометить `# vendored from DSP-GPU/...`.
- `.so` собраны под **cp313** — venv рантайма должен быть cp313 (не cp312; см. спека §1,§7).
- ROCm-контекст создаётся ~1–2 c — Singleton, не пересоздавать в каждом тесте.

## 🚫 Вне P2
Дечирп/куб (§6.1 — отдельно), ФМн-код/коррелятор (P4/§6.2), ЧМ, помехи (P5), движение (P6),
порт наших волн в C++ DSP-GPU (P7).
