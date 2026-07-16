# 🤝 ПЕРЕДАЧА ДЕЛ — body_motion (P3+P4 приняты) → следующий чат: P6

**Дата:** 2026-07-15 · **Статус:** P1,P2,P3,P4,P5 ✅ приняты · остался только **P6**.
**Среда след. сессии:** Linux (Debian, рабочая машина). Пуш P3+P4 — Alex сделает сам.

## 📖 Читать первым в новом чате
1. `MemoryBank/tasks/IN_PROGRESS.md` — актуальный статус всех фаз.
2. `MemoryBank/specs/body_motion_3d_2026-07-15/SPEC.md` — спека (финал), §7·S6 (панель).
3. `MemoryBank/specs/range_scale_dictionary_2026-07-15.md` — дальностные величины (КРИТИЧНО для P5-визуала).
4. `MemoryBank/sessions/2026-07-15_HANDOFF.md` — предыдущая передача (P1/P2/P5, физика ЛЧМ/АМ).
5. `MemoryBank/tasks/TASK_body_motion_p6.md` — следующая задача.
6. Этот файл.

## ✅ Сделано в этой сессии (схема: Кодо таск/сверка → Sonnet код → ревью Кодо)
- **P3** (помехи заград+гребёнка) — был реализован Sonnet ранее, Кодо проверил → ✅ ПРИНЯТО.
  `core/generators/scene_modeler.py` (`SceneModeler`, jammers-only `Scene`), `JammerFlags`
  + optional-спеки в `core/config/scene_config.py`, `demo_body_motion_jammers.py`,
  `tests/test_body_motion_jammers.py` (9 ok). K1/K2/K3 учтены. Вердикт — в TASK_body_motion_p3.md.
- **P4** (мультицель) — Кодо сверил (M1–M4) → выдал Sonnet → Sonnet реализовал → Кодо ✅ ПРИНЯТО.
  - `core/generators/tact_sequence.py`: +`TargetHandle`/`MultiTact`/`MultiTactSequence`
    (координатор N целей, свой seed на цель); **одноцелевой `TactSequence` НЕ тронут** (P1 цел).
  - `core/generators/volume.py`: +`VolumeBuilder.add_shared_noise`, +`iter_multi_cubes`,
    +параметр `add_noise=True` (backward-compat).
  - `core/generators/waveforms/base.py`: +`WaveformSpec.add_noise: bool = True`.
  - `core/generators/waveforms/_pipeline.py`: шум подмешивается только если `spec.add_noise`.
  - `core/generators/__init__.py`: экспорты новых имён.
  - `tests/test_body_motion_multi.py` (`MultiTargetTests`, 6 ok/37 проверок) + `tests/all_test.py`.
  - `demo_body_motion_multi.py` → `graphics/body_motion/p4_multi/` (3 пика целей + помехи, 3 трека).
  - Вердикт (M1–M4) — в TASK_body_motion_p4.md (блок «РЕВЬЮ КОДО»).

## 🔑 Ключевые решения P4 (не переоткрывать)
- **M1 — шум ОДИН раз, не N×.** N целей рендерятся `add_noise=False` (амплитуда всё ещё по
  `snr_db`), суммируются, потом `add_shared_noise` кладёт AWGN 1× поверх суммы. Наивное
  суммирование N зашумлённых объёмов завысило бы мощность шума в N раз.
- **M2 — мультицель РЯДОМ с одноцелевым.** `MultiTactSequence` не наследует/не ломает
  `TactSequence` (тесты P1). Реюз `Kinematics.project`/`MotionModel.propagate`.
- **M3 — когерентная сумма целей = поэлементное `+` массивов `(nx,ny,N)`**, БЕЗ обёртки в
  `SignalSource`/`Scene` (цели живут в waveform-домене; заворачивать ради Composite — лишняя
  косвенность). Помехи P3 применяет ВЫЗЫВАЮЩИЙ код через `SceneModeler.contribute_to` (как в демо P3).
- **M4 — независимость целей** через свой `MotionModel` + свой seed на цель; `doppler_phase`
  уже уникален на цель.
- **Канал шины `tracks`** теперь несёт и `Tact` (одноцель), и `MultiTact` (мультицель) —
  Observer различает по типу. **Учесть в P6** (панель-Observer должна принять оба).

## 🔜 Дальше — P6 (сокет-панель), см. `TASK_body_motion_p6.md`
`Transport` (ZMQ PUB/SUB + MessagePack, fan-out: ZMQ-десктоп + WebSocket-браузер) +
`SceneServer` + Command + Dear PyGui live-панель + веб-дашборд. Образец — `GPUWorkLib/
PyPanelAntennas` (реюз идей color_map/geometry/data-model), но писать СВОЁ. Флаги помех
(`JammerFlags`) едут как команды панели. Закладка ±N плоскостей (блок на сигнал).
**Перед реализацией:** Кодо сверяет таск P6 с кодом (как для P3/P4) → выдаёт Sonnet → ревью.
Ресёрч (Context7/URL): ZMQ PUB/SUB паттерн, msgpack, Dear PyGui live-render.

## ⚠️ Гочи среды (важно на старте)
- **🧹 Удалить перед git:** в этой Windows-сессии остались 0-байтовые `_sync_test.txt` и
  **`.git/index.lock`** (с Linux-песочницы не удалились — права). `.git/index.lock` **блокирует
  git** → `rm -f .git/index.lock _sync_test.txt` перед коммитом.
- **Git не коммичен:** P3+P4 в рабочей копии, но не в git. Alex пушит сам.
- **Линукс-среда:** venv=Python 3.12 (cp312, torch-ROCm). Для чистого прогона тестов нужны
  numpy/scipy/matplotlib (+ruff/mypy для линта).
- **ФС-рассинхрон (был и в этой сессии):** file-edit иногда обрезает/портит кириллицу →
  код писать через **bash heredoc / python write** + `ast.parse` после.
- **Тесты:** 🚫 pytest — только `common.runner.TestRunner`. Прогон: `python3 tests/all_test.py`.
  Известный ложный фейл в песочнице: `MessageBusTests` PermissionError на `/tmp/radar3d_test_
  body_motion_dc/` (каталог `nobody:nogroup` от прошлой сессии) — НЕ регресс, на Linux-машине
  Alex не будет (свежий `/tmp`).

## Как возобновить (Linux, завтра)
1. `rm -f .git/index.lock _sync_test.txt` → `git status` → закоммитить/запушить P3+P4 (Alex).
2. Поднять 6 файлов из «Читать первым» → сказать Кодо «делаем P6».
3. Кодо сверяет TASK_body_motion_p6 с кодом (Transport/шина/панель) + ресёрч → выдаёт Sonnet →
   Sonnet реализует (bash-запись!) → глубокое ревью Кодо (тесты + live-панель + дашборд).
