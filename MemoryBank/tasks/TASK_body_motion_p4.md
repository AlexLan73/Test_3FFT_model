# 🧩 TASK — Body-Motion · P4 (несколько целей: Composite-сцена + TactSequence)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** мульти-цель, координатор тактов.
> **Спека:** [`SPEC.md`](../specs/body_motion_3d_2026-07-15/SPEC.md) (§3,§7·S4).
> **Зависит от:** P3 (сцена с помехами). **Статус:** ✅ ПРИНЯТО (ревью Кодо 2026-07-15).
>
> 🚨 🚫 pytest (`TestRunner`) · 🚫 `.claude/worktrees/**` · существующее НЕ ломать.

---

> 🧭 **Конвенция 3D-визуала (везде):** дальность (range) — по **горизонтали** (пол сцены, вдаль); kx (азимут) — вбок; ky (угол места) — вверх. Динамика: GIF + `--live` окно + `.html`.

## 🎯 Цель P4

Несколько **независимо движущихся** целей в одной сцене (Composite), каждая — со своим
`TargetState` + `MotionModel`. `TactSequence` двигает все состояния и пересобирает объём
каждый такт (Q8 — «всё меняется каждый такт»). Заготовка под трекинг между тактами.

## 🧭 Опорные решения
- Composite: сцена = набор целей + помех, единый `contribute` (реюз P3 `SceneModeler`).
- Каждая цель — свой закон движения (одна cv, другая Markov-дрейф, третья широкий вираж).
- Трекинг «летит» — **между тактами** (гл.4-бис: движение не из одного куба), задел на P6.

## 📦 Что создать / расширить

### 1. `SceneModeler` — довести до мульти-цели (**реюз `Scene`-Composite, A1**)
🔴 **A1:** когерентное суммирование вкладов — это **`Scene.contribute`** (Composite уже готов,
`scene.py`). НЕ дублировать суммирование. `SceneModeler` наполняет `Scene` источниками:
- `targets: list[TargetHandle]` (каждый: `TargetState` + `MotionModel` + waveform-параметры);
- `build_tact(t) -> vol`: для каждой цели `propagate`→`Kinematics`→splat (реюз P2 `VolumeBuilder`),
  добавить как `SignalSource` в `Scene`, + помехи (P3) → `Scene.contribute` даёт когерентную сумму.

### 2. `core/generators/tact_sequence.py` — **расширить** `TactSequence` (создан в P1, A3)
`__iter__`/`__next__` по тактам: двигает **все** состояния, публикует объём + треки
(`DataContext.publish("cube", vol)`, `publish("tracks", states)`). Число тактов/`dt` — из конфига.

### 3. Демо S4 — дополнить `demo_body_motion.py`
2–3 цели с разными законами + помехи → **мульти-трек** 3D в `graphics/body_motion/p4_multi/`:
цветные траектории целей, наложенные на объём.

## ♻️ Реюз (точные пути)
- P1: `core/motion/` (`TargetState`,`MotionModel`,`Kinematics`), `data_context` шина.
- P2: `core/generators/volume.py` (`VolumeBuilder`).
- P3: помехи (`jammers.py`, `jammers_rf.py`), `SceneModeler`.
- Существующая сцена: `core/generators/scene.py` (свериться — реюз Composite, не дублировать).

## ✅ Критерии приёмки
- N целей (N≥2) двигаются независимо; каждая даёт свой пик, треки различимы.
- Когерентное суммирование корректно (комплекс, без потери фаз); входы не мутируются.
- `TactSequence` итерируется, публикует cube+tracks в шину; визуал-Observer обновляется.

## 🧪 Тесты (`TestRunner`)
`MultiTargetTests`: число пиков = числу целей, независимость траекторий, публикация в шину.
Старые целы, ruff/mypy 0.

## 🚫 Границы
Без FFT/квадратов (P5), без сокета/панели (P6). Цели — точечные (протяжённые позже).

---

## 🔎 Сверка Кодо с кодом (2026-07-15, перед реализацией) — ОБЯЗАТЕЛЬНО К УЧЁТУ

Проверил реюз-точки. Найдено 4 узла — учесть в реализации и разобрать на ревью.

- 🔴 **M1 — N-кратный шум (аналог K1 из P3).** `VolumeBuilder.build_from_sample`
  (`core/generators/volume.py`) рендерит поле цели **вместе с шумом** (`render_pipeline`,
  калибровка по `snr_db`, `backend.add_noise`). Вызвать его per-target и сложить →
  шум добавится **N раз** (завышенный/неверный пол). Правильно: **сигналы целей
  суммировать БЕЗ шума, шум добавить ОДИН раз**, затем помехи (P3 `SceneModeler`).
  `VolumeBuilder` no-noise-режима не имеет → расширить **аддитивно, backward-compat**
  (дефолт = с шумом): напр. метод `build_signal_from_sample(...)` (splat без `add_noise`)
  или флаг `with_noise: bool = True`. Сверить, как `render_pipeline`/`NumpyBackend`
  добавляет шум (`core/generators/waveforms/_pipeline.py`, `backends/numpy_backend.py`) —
  выбрать минимально инвазивный путь. Старые вызовы (`build`/`build_from_sample` в P2/P5/демо)
  НЕ менять по поведению.
- 🔴 **M2 — `TactSequence` одноцелевой (не ломать P1).** `core/generators/tact_sequence.py`:
  `TactSequence(Iterator[Tact])` держит ОДНУ цель (`initial: TargetState`, `model`),
  покрыт тестами P1 (`tests/test_body_motion.py`). Расширять его на мультицель = **риск
  регресса P1**. Рекомендация: **НЕ трогать `TactSequence`**, добавить рядом
  `MultiTactSequence` (или координатор) — держит `list[(TargetState, MotionModel)]` (+ свой
  `rng`/seed на цель, чтобы `MarkovDrift` независим), на такте двигает все, публикует
  `cube`+`tracks` в шину. Реюз `Kinematics.project` и `MotionModel.propagate` как есть.
  Если всё же расширять `TactSequence` — обосновать и доказать зелёность P1-тестов.
- 🟡 **M3 — два механизма продукции, свести АДДИТИВНО (A1: не плодить Composite).**
  Цели идут через `VolumeBuilder` (waveform-render, `NumpyBackend`), помехи P3 — через
  `Scene/SignalSource` (`SceneModeler`). Оба дают `(nx,ny,N)` complex → когерентная сумма =
  поэлементное сложение массивов. `build_tact(states)` = `Σ_i signal(state_i)` + шум(1×) +
  вклад помех (реюз `SceneModeler`: помехи можно взять из `scene.contribute`, шум — отдельно).
  **Не** строить единый `Scene`-Composite для целей+помех, если проще сложить массивы —
  цели живут в waveform-домене, не в `SignalSource`. Обосновать выбор на ревью.
- 🟡 **M4 — Доплер/независимость на цель уже есть.** `KinematicsSample.doppler_phase`
  уникален на цель (идёт в `spec.phase`), фазы не путаются. Достаточно дать каждой цели
  свой `MotionModel` (`ConstantVelocity`/`MarkovDrift`/`CoordinatedTurn`/`WeavingManeuver`
  — все в `core.motion`) и свой seed.
- **Тесты (`MultiTargetTests`):** «N пиков = N целей» проверять на **разнесённых** R и/или
  углах (в сыром домене без FFT близкие по дальности цели сольются — задавать разные R/kx/ky).
  Пики — по энергии-по-дальности (окна разных R) и/или по угловой карте `FFT2` (разные kx,ky,
  как в демо P3). Независимость треков: разошлись по позиции за N тактов. Публикация в шину:
  подписать тестовый Observer, проверить приход `cube`+`tracks`.
- **Запись файлов:** через bash heredoc / `python - <<'EOF'` + `ast.parse` после (ФС-гоча
  среды). Тесты: `python3 tests/all_test.py`. 🚫 pytest · 🚫 `.claude/worktrees/**` · НЕ коммитить.

---

## ✅ РЕВЬЮ КОДО (2026-07-15) — ПРИНЯТО

Реализовано Sonnet-агентом. Файлы: `tact_sequence.py` (+`TargetHandle`/`MultiTact`/
`MultiTactSequence`, одноцелевой `TactSequence` НЕ тронут), `volume.py`
(+`add_shared_noise`, `iter_multi_cubes`, параметр `add_noise=True`), `waveforms/base.py`
(+`WaveformSpec.add_noise`), `waveforms/_pipeline.py` (гейт шума), `__init__.py` (экспорты),
`tests/test_body_motion_multi.py` (+`all_test.py`), `demo_body_motion_multi.py`.

**Сверка находок:**
- **M1** ✓ — флаг `add_noise` протянут `WaveformSpec`→`render_pipeline`; цели рендерятся
  без шума, `add_shared_noise` кладёт AWGN ОДИН раз в `iter_multi_cubes`. Тест
  `noise_added_once_not_n_times`: var шума ~1.0 при N=1 и N=4 (не 4×). Дефолт `True` —
  старое поведение цело (P2/P5 не затронуты).
- **M2** ✓ — `TactSequence` не изменён, `MultiTactSequence` добавлен рядом; регресс P1
  (`tests/test_body_motion.py`) — 0 (MotionModel/Kinematics зелёные).
- **M3** ✓ — когерентная сумма = поэлементное сложение `(nx,ny,N)` напрямую, без обёртки
  в `SignalSource` (обосновано в докстринге `iter_multi_cubes`); помехи P3 применяет
  вызывающий код через `SceneModeler.contribute_to` (как в демо P3).
- **M4** ✓ — свой seed/ГСЧ на цель; тест `targets_move_independently`: траектория цели A
  бит-в-бит одинакова при любой цели B (модель/seed).

**Проверки:** `MultiTargetTests` **6 ok/0 fail** (37 проверок); весь набор зелёный, КРОМЕ
1 фейла `MessageBusTests::test_data_context_composes_bus_and_keeps_save_load` — это **среда,
не P4**: каталог `/tmp/radar3d_test_body_motion_dc/` принадлежит `nobody:nogroup` (создан до
сессии), `tests/test_body_motion.py` P4 НЕ трогал (git diff пуст), свежие записи в `/tmp`
работают. На машине Alex не воспроизведётся. ruff/mypy — 0. Демо → 3 файла в `p4_multi/`
(угловая карта: 3 пика целей + заград/гребёнка; мульти-трек: 3 независимые траектории).

**Замечаний-блокеров нет.** Мелочи (не блокеры):
1. Канал `tracks` теперь несёт и `Tact` (одноцель), и `MultiTact` (мультицель) — Observer
   должен различать по типу. Задокументировано; учесть в P6 (панель-Observer).
2. 🧹 Среда оставила 0-байтовые `_sync_test.txt` и `.git/index.lock` (с Linux-стороны не
   удаляются — права). **`.git/index.lock` удалить перед любым git-коммитом**, иначе git
   откажет.
