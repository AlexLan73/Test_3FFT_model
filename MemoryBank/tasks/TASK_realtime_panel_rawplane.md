# TASK — Реалтайм-панель: план сырья (посадка под GPU)

> Спека: `specs/realtime_panel_2026-07-19.md` (§2.2, §3 «Этап B/C/D», §4.1 два плана, §7).
> Старт: 2026-07-20. Ветка `main`. Делегировано Sonnet, ревью Кодо.

## Зачем

Построить **план сырья** (§4.1) — транспорт-слой, куда ПОТОМ воткнётся генерация на GPU (другой чат):
источник → `RawQueue` (consume-and-drop, не копим ~10 ГБ/такт) → обработка забирает. Обработка сырья
(фронтенд по `sig` + ядро) и генерация на GPU — НЕ здесь (GPU-чат). Здесь только транспорт сырья, torch-free.

## Что делаем (Sonnet)

1. `core/runtime/raw_queue.py` — `RawFrame` (VO: index/cube/sig), `RawQueue` (bounded consume-and-drop,
   thread-safe, `dropped`-счётчик, `on_drop`-хук под архив блоками).
2. `core/runtime/raw_source.py` — `RawCubeSource` (ABC Strategy), `FileSource` (Этап D: .npy потактно
   с диска через `run(sink, stop)` + `iter_frames`). SocketSource — Этап C, вне задачи (заметка).
3. Реэкспорт в `core/runtime/__init__.py`. Тесты `tests/test_raw_queue.py` (+ регистрация в `all_test.py`).
4. README-секция про рантайм (публичный API + мини-пример + ссылка на спеку §7).

## Приёмка (Кодо, после Sonnet) — ✅ ПРИНЯТО 2026-07-20

- [x] `git diff` прочитан, соответствует ТЗ; чужие файлы (transport/codec/scene_server/generators/
      models/example/server/live_demo/panel_publisher/web) НЕ тронуты (только README/__init__/all_test + 3 новых).
- [x] `tests/test_raw_queue.py` (`RawQueueTests` 5 · `FileSourceTests` 3) + `tests/all_test.py` —
      прогнал сам: **весь бэкенд 0 fail**.
- [x] Независимый smoke: `RawQueue(maxsize=2)` дропнул `[0,1,2]` (dropped=3), остались свежие `[3,4]`,
      `on_drop` видел все три; `FileSource` читает 3 temp-файла по порядку с целостностью данных; stop-до-старта → 0.
- [x] Набор в `all_test.py`; README-секция «Realtime-панель (core.runtime)» корректна (таблица двух планов + пример).

## Дальше

GPU-чат: генерация сырья → `RawQueue.put(RawFrame(i, cube, sig))`. Этап B: `SignalFrontend[sig]` +
ядро (из `RawQueue.get`) → `PanelPublisher.push_tick`. Хук `on_connect` (§4.4). Канал `start/stop` (§4.2).
