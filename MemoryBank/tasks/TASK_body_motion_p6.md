# 🧩 TASK — Body-Motion · P6 (сокет-панель: ZMQ Observer + Dear PyGui + закладка ±N)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** рантайм-транспорт + live-панель.
> **Спека:** [`SPEC.md`](../specs/body_motion_3d_2026-07-15/SPEC.md) (§4,§5,§6,§7·S6).
> **Образец панели:** `E:\C++\GPUWorkLib\PyPanelAntennas\Examples\` (Dear PyGui, F8) — **референс,
> изучаем подход и пишем СВОЁ**, не вендорим/не копируем 1:1.
> **Зависит от:** P1–P5 (сцена + квадраты). **Статус:** ✅ ПРИНЯТО (ревью Кодо 2026-07-16).
>
> 🚨 🚫 pytest (`TestRunner`) · 🚫 `.claude/worktrees/**` · существующее НЕ ломать.

---

> 🧭 **Конвенция 3D-визуала (везде):** дальность (range) — по **горизонтали** (пол сцены, вдаль); kx (азимут) — вбок; ky (угол места) — вверх. Динамика: GIF + `--live` окно + `.html`.

## 🎯 Цель P6

Живая **панель управления** сценой: продюсер (сцена) публикует кадры через сокет, панель —
**Observer** — реагирует и рисует. Управление: add/remove цель, вкл/выкл помеху, шаг такта,
разворачиваемая закладка «окрестность объекта ±N плоскостей». **Кросс-язычно (py↔C++)** —
контракт сообщений язык-нейтрален (S1-совместимо), чтобы завтра C++/GPU-продюсер встал без
смены протокола/панели.

## 🧭 Опорные решения (спека/ревью)
- **Транспорт: ZMQ PUB/SUB + MessagePack** (§4/§6) — нативный Observer, кросс py↔C++.
- **Панель: Dear PyGui** (60 fps); `PyPanelAntennas` — **образец** (F8): изучаем подход
  (color_map/geometry/data-model) и пишем **своё** под нашу модель квадратов 16×16 + ZMQ; не копируем.
- 🟡 **A7 — два уровня Observer, не путать:** `MessageBus` (P1) — **внутрипроцессный** (in-process
  publish→notify); `Transport` (ZMQ) — **межпроцессный** мост. Граница: продюсер(процесс1) →
  `Transport` PUB → ZMQ → (процесс2) `Transport` SUB в треде → кладёт в **локальный** `MessageBus`/queue
  → GUI-Observer рисует. На **каждой** стороне свой `MessageBus`; `MessageBus` через процессы НЕ ходит.
- 🟢 **ДВА ФРОНТА, ОДИН ДВИЖОК (решение Alex 2026-07-15):** продюсер публикует **один раз**,
  подключаются **оба** фронта: (а) **веб-дашборд** (Three.js/HTML в браузере) и (б) **десктоп**
  (Dear PyGui). Оба — Observer'ы над `Transport`. `Transport` — **fan-out** (Strategy/Composite):
  `ZmqTransport` (десктоп, прямой SUB) + `WebSocketTransport` (браузер — raw-ZMQ в браузере НЕ
  работает, нужен WS-мост/шлюз). MessagePack декодится и в python, и в JS. Ядро/движок не дублируется.
- Кросс-язычность: не pickle; схемы MessagePack + модель `Field/Cell/Element` — единый контракт.

## 📦 Что создать

### 1. `core/runtime/transport.py` — `Transport(Protocol)` (Strategy) + fan-out
`publish(topic, obj)` / `subscribe(topic, cb)`. Реализации:
- `ZmqTransport` — PUB/SUB поверх `pyzmq` + `msgpack` (десктоп-фронт, прямой SUB).
- `WebSocketTransport` — WS-мост для браузера (raw-ZMQ в браузере не работает); тот же
  `msgpack`-кодек. Лёгкий шлюз ZMQ→WS (или продюсер публикует в оба).
- `FanOutTransport` (Composite) — публикует **один раз** в оба канала → «один движок, два фронта».
Кодек — `codec.py` (схемы язык-нейтральны, py↔JS). Обёртка тонкая → подмена/добавление
транспорта без смены логики сцены.

### 2. `core/runtime/scene_server.py` — `SceneServer` (издатель + Command)
Крутит `TactSequence` (P4), на каждый такт **публикует** `cube`/`squares`/`tracks` через
`Transport`. Принимает **команды** (Command pattern) от панели по обратному каналу:
`AddTarget`,`RemoveTarget`,`SetMotion`,`EnableJammer{…}`,`Step(dt)`,`SetNeighborPlanes(N)`.

### 3. `core/runtime/commands.py` — `Command` (Command pattern)
Сериализуемые команды (msgpack) панель→сервер; применяются к сцене на следующем такте.

### 4. `core/graphics/panel/` — Dear PyGui панель (Observer) — **своё по образцу F8**
- Пишем **своё**, подсматривая структуру `PyPanelAntennas/Examples/` (образец, не копия):
  cmap (аналог `color_map.py`), раскладка rect/circle/hit-test (аналог `geometry.py`), дата-модель
  `Field/Cell/Element` + lerp-анимация (аналог `data_models.py`) — под нашу сетку квадратов 16×16.
- `panel_app.py` — окна: поле квадратов 16×16 (live), контролы (цели/помехи/шаг), лог.
- **Приём через наш `Transport`** (ZMQ SUB) в треде → queue → GUI дренит (реюз паттерна
  самоанимации `data_models`).
- **Закладка «окрестность ±N» (§5):** объект в плоскости K → показать `K−2…K+2` (N дефолт 5,
  регулируемо). Блок на сигнал (кол-во блоков = кол-ву сигналов), **3 ряда**: теплокарта
  (2D/3D-FFT) → место в квадрате → описание точек (токены+подписи). Заград — отдельный блок
  с пометкой углов.

### 4b. `web/` — веб-дашборд (Three.js/HTML, Observer через WebSocket)
Самодостаточный HTML + JS (Three.js): 3D-сцена (дальность в горизонте) + квадраты 16×16 +
контролы (play/пауза/таймлайн, цели/помехи), закладка ±N. Подписка на `WebSocketTransport`,
декод `msgpack` в браузере. Прообраз — интерактивный вьюер P1 (реальные данные, вращение+play).
Открывается в любом браузере, без установки.

### 5. Демо S6 — `demo_body_motion_panel.py` (Composition Root)
Запуск: `SceneServer` (продюсер, numpy) + панель (Observer) на localhost через ZMQ. Управление
из панели меняет сцену вживую. Плюс plotly 3D-обзор — по кнопке снимок.

## ♻️ Реюз (точные пути)
- Панель-**образец** (референс, не копия): `GPUWorkLib/PyPanelAntennas/Examples/`
  (`color_map`,`geometry`,`data_models`,`main`) — изучаем подход, пишем своё.
- Шина/Observer: `core/data_context/message_bus.py` (P1).
- Сцена/кадры: `TactSequence` (P4), `SquareView`/квадраты (P5).
- Зависимости: `pyzmq`, `msgpack`, `dearpygui` — pip на Windows (в `pyproject` optional-группа `panel`).

## ✅ Критерии приёмки
- Продюсер публикует, панель-Observer **обновляется** на каждый кадр (без прямого вызова рендера).
- Команды панель→сервер работают: add/remove цель, вкл/выкл помеха, шаг такта, `N` закладки.
- Закладка ±N: блоки по сигналам, 3 ряда, заград отдельным блоком с углами.
- Контракт msgpack язык-нейтрален (нет pickle/py-специфики) — задокументированы схемы для C++.
- Кросс-платформа: сменив продюсера (numpy→C++/GPU в будущем), панель/протокол не меняются.

## 🧪 Тесты (`TestRunner`)
`TransportTests` (publish→subscribe roundtrip, msgpack кодек), `CommandTests` (применение к сцене),
`PanelModelTests` (дата-модель/закладка ±N — без GUI, логика). Dear PyGui-часть — под `SkipTest`,
если библиотеки нет. Старые целы, ruff/mypy 0.

## 🚫 Границы
Полный детектор/токенизатор — не здесь. C++/GPU-продюсер — P7 (порт). Здесь python-референс +
язык-нейтральный контракт.

---

## 🔎 Сверка Кодо с кодом/средой (2026-07-16, перед реализацией) — ОБЯЗАТЕЛЬНО К УЧЁТУ

Проверил реюз-точки и среду. Реюз-классы есть: `MessageBus` (`core/data_context/message_bus.py`),
`SquareView` (`core/graphics/square_view.py`), `MultiTactSequence`/`TactSequence`
(`core/generators/tact_sequence.py`), `SceneModeler` (`core/generators/scene_modeler.py`),
`JammerFlags`/`SceneConfig` (`core/config/scene_config.py`). `core/runtime/` и `web/` — создать с нуля.

- 🔴 **N1 — ZMQ PUB/SUB «slow joiner».** PUB отбрасывает сообщения, отправленные ДО того, как SUB
  подключился и подписался. Для live-панели это ок (поздний подписчик пропускает старые кадры).
  Но **тесты roundtrip** станут флейки: нельзя `pub.send` сразу после `sub.connect`. В `TransportTests`
  использовать барьер: bind PUB → connect SUB → **poll-цикл** до первого сообщения (или `inproc://`
  + короткий `sleep`/poller). Командный канал (панель→сервер) лучше **PUSH/PULL** (или REQ/REP) —
  там потеря недопустима, PUB/SUB не годится. Обосновать выбор.
- 🔴 **N2 — msgpack ЯЗЫК-НЕЙТРАЛЬНО (критерий приёмки «контракт для C++»).** 🚫 НЕ `pickle`,
  🚫 НЕ `msgpack-numpy`-ext (это py-специфичное расширение, C++ его не разберёт). Комплексные
  массивы кодировать **явной схемой**: `{"topic":..., "tact":int, "shape":[nx,ny,N], "dtype":"complex64",
  "data": <raw bytes>}` (или `real`/`imag` раздельными bytes) — фикс endianness (little), задокументировать.
  `codec.py` = единственное место (де)кодирования, схемы описать в докстринге для C++/JS. JS-декод
  (веб-дашборд) обязан читать ту же схему (проверить хотя бы структуру в браузере).
- 🟡 **N3 — два уровня Observer (A7), не путать.** `MessageBus` (P1) — ВНУТРИпроцессный;
  `Transport` (ZMQ/WS) — МЕЖпроцессный. SUB крутится в треде → кладёт в **локальный** queue/`MessageBus`
  → GUI дренит. `MessageBus` через процессы НЕ ходит. На каждой стороне свой.
- 🟡 **N4 — `Transport(Protocol)` + `FanOutTransport` (Composite), publish ОДИН раз.** raw-ZMQ в
  браузере не работает → `WebSocketTransport` = шлюз ZMQ→WS (python: лёгкий `websockets`-сервер
  ре-паблишит те же msgpack-кадры) ИЛИ `SceneServer` публикует в оба через `FanOutTransport`.
  Тонкая обёртка `publish(topic,obj)`/`subscribe(topic,cb)` — подмена транспорта без смены логики сцены.
- 🟡 **N5 — Dear PyGui headless НЕ проверить в среде.** Логику панели вынести в **GUI-free**
  `panel_model.py` (дата-модель `Field/Cell/Element`, выбор ±N плоскостей, lerp) — она покрывается
  `PanelModelTests`. `panel_app.py` (обвязка dearpygui) — тонкая, импорт dearpygui под `try/except`
  → `SkipTest`, если библиотеки/дисплея нет. НЕ смешивать логику с GUI (тестируемость + SRP).
- 🟡 **N6 — образец `E:\C++\GPUWorkLib\PyPanelAntennas` в среде НЕДОСТУПЕН** (вне воркспейса).
  Писать своё по описанию из этого таска (cmap / rect-hit-test / `Field-Cell-Element` + lerp), без вендоринга.
- **Среда/тесты:** `msgpack`+`pyzmq` поставить (`pip install --break-system-packages msgpack pyzmq`),
  прогнать `TransportTests`/`CommandTests`/`PanelModelTests` headless (ZMQ на `tcp://127.0.0.1:*` или
  `inproc://` работает без дисплея). `dearpygui` не ставить (нет дисплея) → его тесты под `SkipTest`.
  Опциональная группа в `pyproject.toml`: `panel = ["pyzmq","msgpack","dearpygui","websockets"]`.
- **Ресёрч перед кодом (правило 00):** Context7 по `pyzmq` (PUB/SUB, PUSH/PULL, poller), `msgpack`
  (raw bytes, streaming Unpacker), `dearpygui` (тред→queue→GUI drain, texture/draw) — свежие API.
- **Запись файлов:** bash heredoc / `python - <<'EOF'` + `ast.parse` после (ФС-гоча среды).
  🚫 pytest · 🚫 `.claude/worktrees/**` · существующее НЕ ломать · НЕ коммитить.

---

## ✅ РЕВЬЮ КОДО (2026-07-16) — ПРИНЯТО

Реализовано Sonnet-агентом (после Context7-ресёрча pyzmq/msgpack/dearpygui). Файлы:
`core/runtime/` (`codec.py`,`transport.py`,`commands.py`,`scene_server.py`,`__init__.py`),
`core/graphics/panel/` (`panel_model.py` GUI-free + `panel_app.py` тонкая обвязка dearpygui),
`web/` (`index.html`,`app.js`,`codec.js`), `demo_body_motion_panel.py`, `tests/test_runtime.py`
(+`all_test.py`), `pyproject.toml` (+группа `panel`).

**Сверка находок:**
- **N1** ✓ — данные `cube`/`squares`/`tracks` = PUB/SUB (slow-joiner drop желателен для live),
  команды = PUSH/PULL (topic `cmd`, не теряются); тесты roundtrip через poll-барьер. REQ/REP
  отклонён с обоснованием.
- **N2** ✓ — `codec.py` единственная точка, ЯВНАЯ язык-нейтральная схема (`{topic,tact,kind,payload}`,
  массивы = `{shape,dtype,data:raw LE bytes}`, complex=interleaved). 🚫 pickle/msgpack-ext. `web/codec.js`
  повторяет 1:1. Проверил: роундтрип complex64 точный (`array_equal=True`), `node --check` JS ок.
- **N3** ✓ — `MessageBus` (внутрипроц.) не ходит через процессы; SUB в треде → `queue.Queue` → GUI дренит.
- **N4** ✓ — `FanOutTransport.publish` шлёт 1 раз в оба (ZMQ+WS); проверено тестом + e2e.
- **N5** ✓ — `panel_model.py` без GUI-импортов, `PanelModelTests` 8 ok; `panel_app.py` под
  `try/except`→`SkipTest` (headless).
- **N6** ✓ — образец недоступен, написано своё.

**Проверки:** весь набор зелёный (`Transport` 7 · `Command` 7 · `SceneServerStep` 2 · `PanelModel` 8 ·
`PanelApp` skip; регрессов нет, включая `MessageBusTests` 4 ok). ruff 0; mypy 0 на новых файлах.

**Что НЕ проверяемо headless (ожидаемо):** живое окно Dear PyGui и рендер Three.js в браузере —
провод-контракт проверен e2e (Sonnet: python-WS-клиент декодит реальные кадры) + codec-роундтрип (Кодо).

**Правки Кодо на ревью:**
- 🔴→✅ **gitignore-ловушка:** строка `graphics/` (неякорёная) прятала новый пакет
  `core/graphics/panel/` от git (`git check-ignore` подтверждал). Починил: `graphics/` → `/graphics/`
  (игнорим только корневой вывод, пакет-исходник трекается). Проверено.

**Мелочи (не блокеры):**
1. `SceneServer` НЕ реюзит `MultiTactSequence` (живой мутируемый по командам список целей vs
   фиксированный итератор) — обосновано в докстринге, реюз примитивов `Kinematics`/`MotionModel`
   + VO `Tact`/`MultiTact`. 3 строки шумовой логики (M1/M3) продублированы осознанно — ок.
2. 🟡 Преждесуществующий (НЕ P6) mypy-хвост `core/graphics/sampling.py:59` (всплывает транзитивно
   через импорт `SquareView`) — отдельная мелкая правка на потом.
