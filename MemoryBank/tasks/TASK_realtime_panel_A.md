# TASK — Реалтайм-панель, Этап A (PanelPublisher + live_demo)

> Спека: `specs/realtime_panel_2026-07-19.md` (§1, §2.1, §3 «Этап A», §4.1).
> Старт: 2026-07-20. Ветка `main`.

## Контекст двух чатов (2026-07-20)

- **Этот чат (база/demo, БЕЗ GPU):** развязка источник↔сервер — `PanelPublisher`/`TickLog`/`Tick`
  + `demo/ex4_flight/live_demo.py`. Вариант 2 (готовые примитивы), доказать реалтайм end-to-end.
- **Другой чат (GPU):** обработка сырых данных на GPU (генерация/детекция куба). Не пересекаемся:
  `SceneServer` (cube/squares/tracks/tokens) — их территория, я его не трогаю.

## Что делаем (Этап A, делегировано Sonnet)

1. `core/runtime/panel_publisher.py` — `Tick` (frozen VO) · `TickLog` (append-only, храним всё,
   опц. `cap`) · `PanelPublisher` (DI на `Transport`: `start/push_meta/push_tick/republish_meta/close`).
   Реэкспорт из `core/runtime/__init__.py`.
2. `demo/ex4_flight/live_demo.py` — детерминированная летящая точка → минимальный валидный `tick`
   (§2.1: truth/pts/trk/след kTrail=8, sl=[]), meta с камерой `Projection.as_js()`. Реалтайм-цикл
   с `time.sleep`, `republish_meta` каждые ~30 тактов (поздний клиент, on_connect §4.4 — позже).
3. Тесты (TestRunner): `PanelPublisherTests`/`TickLogTests` в `tests/test_runtime.py` (fake-transport,
   roundtrip codec, «храним всё», порядок) + `LiveDemoTests` в `demo/tests/test_live_demo.py`.
   Регистрация в `tests/all_test.py` и `demo/tests/all_demo_test.py`.

## Приёмка (Кодо, после Sonnet) — ✅ ПРИНЯТО 2026-07-20

- [x] `git diff` прочитан, соответствует ТЗ (без отсебятины, чужие файлы transport/codec/scene_server/
      example/server/web НЕ тронуты).
- [x] `tests/test_runtime.py` (`TickLogTests` 3 · `PanelPublisherTests` 6) + `tests/all_test.py` —
      прогнал сам: **весь бэкенд 0 fail**.
- [x] `demo/tests/test_live_demo.py` (7 ok) + `demo/tests/all_demo_test.py` — **ВСЁ ЗЕЛЁНОЕ**.
- [x] Независимый smoke: fake-transport поймал meta+3×tick, позиции цели меняются, TickLog растёт,
      codec roundtrip всех, `republish_meta` шлёт meta повторно.
- [x] Новые наборы зарегистрированы в `tests/all_test.py` и `demo/tests/all_demo_test.py`.
- [ ] ⏳ Визуальная проверка в браузере (`live_demo.py` + web/) — за Alex (реалтайм на уровне данных доказан).

## Дальше (после A)

Этап B — сырой куб → фронтенд по `sig` → ядро → tick (реюз `core`; часть — GPU-чат).
Хук `on_connect` в `WebSocketTransport` (§4.4, поздний клиент → реплей meta + лог).
