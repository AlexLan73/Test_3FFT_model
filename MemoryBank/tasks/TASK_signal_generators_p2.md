# 🧩 TASK — Генераторы сигналов · P2 (gpu_libs loader + GPU-smoke + HipBackend + сверка)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** обвязка готовых GPU-моделей DSP-GPU.
> **Спека:** [`specs/signal_generators_2026-07-13.md`](../specs/signal_generators_2026-07-13.md) (§2.2,§2.2.1,§4.3,§8).
> **Зависит от:** P0+P1 (`SignalField`, `NumpyBackend`, волны, тесты). **Статус:** ⬜ НЕ НАЧАТО.
>
> 🚨 🚫 pytest (только `TestRunner`) · 🚫 `.claude/worktrees/**`.
> 🐧 **Только Linux+ROCm.** Всё, что GPU, — под `SkipTest`, если нет `.so`/ROCm (Windows/CI).

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
