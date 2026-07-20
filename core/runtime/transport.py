"""Transport -- межпроцессный мост продюсер<->панель (Strategy, P6).

🟡 N3 (граница Observer, не путать): `core.data_context.MessageBus` (P1) --
ВНУТРИпроцессный Subject/Observer (один процесс, синхронный `publish`->`notify`).
`Transport` здесь -- МЕЖпроцессный канал: продюсер (процесс 1) вызывает
`publish(topic, tact, payload)`, приёмник (процесс 2) получает данные в СВОЁМ
фоновом треде (`ZmqTransport`/`WebSocketTransport` держат SUB/PULL-сокет в
`threading.Thread`) и кладёт их в локальную очередь/`MessageBus` этого процесса --
`MessageBus` НИКОГДА не ходит через границу процессов, только `Transport`.

🔴 N1 (slow joiner + надёжность команд): PUB отбрасывает всё, отправленное ДО того,
как SUB подключился/подписался ("slow joiner", доки pyzmq howto/logging.md) -- для
живого превью кадров это приемлемо (поздний подписчик просто не видит старые
кадры, ему нужен ТОЛЬКО текущий такт). Но для канала команд (панель->сервер)
потеря недопустима -- одна пропущенная `AddTarget` меняет смысл сцены незаметно
для оператора. Поэтому здесь ДВА разных сокет-паттерна под ОДНИМ интерфейсом
`publish/subscribe`, различаемых зарезервированным именем канала `CMD_TOPIC`:
  - топик `!= CMD_TOPIC`  -> ZMQ PUB/SUB (данные такта: cube/squares/tracks) --
    отбрасывание старых кадров при позднем подключении -- ЖЕЛАТЕЛЬНОЕ поведение;
  - топик `== CMD_TOPIC`  -> ZMQ PUSH/PULL -- PUSH СТАВИТ сообщения в очередь
    (до HWM), а не роняет их, если PULL ещё не подключился -- команда не теряется.
  REQ/REP здесь НЕ взят: REQ/REP синхронно блокирует отправителя до ответа
  (строгий lock-step "запрос-ответ"), а команды панели -- fire-and-forget
  (применяются на СЛЕДУЮЩЕМ такте сервера, ответ не нужен) -- PUSH/PULL проще и
  не блокирует GUI-поток панели на send().

🟡 N4 (`FanOutTransport`, Composite): продюсер публикует ОДИН раз в
`FanOutTransport.publish(...)`, тот форвардит в КАЖДЫЙ вложенный транспорт --
"один движок сцены, два фронта" (десктоп-панель через `ZmqTransport` + браузер
через `WebSocketTransport`) без дублирования цикла продюсера.
"""
from __future__ import annotations

import queue
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from . import codec

CMD_TOPIC = "cmd"   # зарезервированное имя канала команд -- см. докстринг модуля (N1)

Callback = Callable[[str, int, object], None]


class Transport(Protocol):
    """Strategy: `publish(topic, tact, payload)` / `subscribe(topic, callback)`.

    Тонкий контракт (2 глагола) -- подмена/добавление реализации не требует
    менять логику сцены/панели (`SceneServer`/`PanelApp` работают только через
    этот протокол, см. критерий приёмки TASK P6 "кросс-платформа").
    """

    def publish(self, topic: str, tact: int, payload: object) -> None: ...

    def subscribe(self, topic: str, callback: Callback) -> None: ...


class ZmqTransport:
    """ZMQ-реализация: PUB/SUB (данные) + PUSH/PULL (команды, `CMD_TOPIC`), см. N1.

    Роли задаются набором адресов -- сервер (продюсер) обычно ПРИВЯЗЫВАЕТ оба
    сокета (`data_bind`+`cmd_bind`, он один и стабилен), панель(и) ПОДКЛЮЧАЮТСЯ
    (`data_connect`+`cmd_connect`, их может быть несколько). Каждая сторона
    задаёт только то, что ей нужно (панель может не иметь `data_bind`, сервер --
    `data_connect`).

    Приёмный сокет (SUB или PULL) живёт ИСКЛЮЧИТЕЛЬНО в своём треде (`_data_loop`/
    `_cmd_loop`) -- ZMQ-сокеты не потокобезопасны для одновременного использования
    из нескольких тредов; `subscribe()` из вызывающего (например, GUI) треда НЕ
    трогает сокет напрямую, а кладёт имя топика в `queue.Queue` (`_pending_*`),
    которую сам поток-приёмник вычитывает перед каждым `poll()`.
    """

    # Типы zmq-сокетов/контекста -- `Any` намеренно: `pyzmq` опциональна (optional-
    # группа `panel`), импортируется ЛОКАЛЬНО внутри `__init__` (не на уровне модуля),
    # поэтому статическая типизация конкретными классами `zmq.Socket`/`zmq.Context`
    # потребовала бы безусловного `import zmq` в шапке файла -- вернула бы модуль в
    # разряд обязательных зависимостей. `core.runtime.transport` обязан импортироваться
    # ДАЖЕ без pyzmq (например, чтобы `WebSocketTransport`/`FanOutTransport` работали
    # сами по себе) -- см. критерий приёмки TASK P6 "headless без дисплея".
    def __init__(
        self,
        data_bind: str | None = None,
        data_connect: str | None = None,
        cmd_bind: str | None = None,
        cmd_connect: str | None = None,
        context: Any | None = None,
        poll_timeout_ms: int = 100,
    ) -> None:
        import zmq  # локальный импорт -- pyzmq опциональна (optional-группа `panel`)

        if data_bind and data_connect:
            raise ValueError("ZmqTransport: задайте либо data_bind, либо data_connect, не оба")
        if cmd_bind and cmd_connect:
            raise ValueError("ZmqTransport: задайте либо cmd_bind, либо cmd_connect, не оба")

        self._zmq: Any = zmq
        self._ctx: Any = context or zmq.Context.instance()
        self._poll_timeout_ms = poll_timeout_ms

        self._pub: Any = None
        if data_bind:
            self._pub = self._ctx.socket(zmq.PUB)
            self._pub.bind(data_bind)
        self._data_connect = data_connect

        self._push: Any = None
        if cmd_connect:
            self._push = self._ctx.socket(zmq.PUSH)
            self._push.connect(cmd_connect)

        # PULL -- биндится СРАЗУ (как PUB выше), а не лениво в subscribe(): сервер
        # должен уметь сообщить фактический bound_cmd_endpoint() (например при
        # cmd_bind="tcp://127.0.0.1:*") ДО первого subscribe(CMD_TOPIC, ...).
        self._pull: Any = None
        if cmd_bind:
            self._pull = self._ctx.socket(zmq.PULL)
            self._pull.bind(cmd_bind)

        self._data_callbacks: dict[str, list[Callback]] = defaultdict(list)
        self._cmd_callbacks: list[Callback] = []
        self._callbacks_lock = threading.Lock()

        self._pending_data_subs: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._data_thread: threading.Thread | None = None
        self._cmd_thread: threading.Thread | None = None
        self._sub: Any = None   # создаётся лениво в _ensure_data_thread (клиент знает свой адрес сам)

    # -- publish ----------------------------------------------------------
    def publish(self, topic: str, tact: int, payload: object) -> None:
        raw = codec.encode(topic, tact, payload)
        if topic == CMD_TOPIC:
            if self._push is None:
                raise RuntimeError("ZmqTransport: cmd_connect не задан -- publish(CMD_TOPIC) недоступен")
            self._push.send(raw)
        else:
            if self._pub is None:
                raise RuntimeError("ZmqTransport: data_bind не задан -- publish(data) недоступен")
            self._pub.send_multipart([topic.encode("utf-8"), raw])

    # -- subscribe ----------------------------------------------------------
    def subscribe(self, topic: str, callback: Callback) -> None:
        if topic == CMD_TOPIC:
            if self._pull is None:
                raise RuntimeError("ZmqTransport: cmd_bind не задан -- subscribe(CMD_TOPIC) недоступен")
            with self._callbacks_lock:
                self._cmd_callbacks.append(callback)
            self._ensure_cmd_thread()
        else:
            if self._data_connect is None:
                raise RuntimeError("ZmqTransport: data_connect не задан -- subscribe(data) недоступен")
            with self._callbacks_lock:
                self._data_callbacks[topic].append(callback)
            self._pending_data_subs.put(topic)
            self._ensure_data_thread()

    # -- фоновые треды ----------------------------------------------------------
    def _ensure_data_thread(self) -> None:
        if self._data_thread is not None:
            return
        sub = self._ctx.socket(self._zmq.SUB)
        sub.connect(self._data_connect)
        self._sub = sub
        self._data_thread = threading.Thread(target=self._data_loop, daemon=True)
        self._data_thread.start()

    def _ensure_cmd_thread(self) -> None:
        if self._cmd_thread is not None:
            return
        self._cmd_thread = threading.Thread(target=self._cmd_loop, daemon=True)
        self._cmd_thread.start()

    def _data_loop(self) -> None:
        poller = self._zmq.Poller()
        poller.register(self._sub, self._zmq.POLLIN)
        while not self._stop.is_set():
            while not self._pending_data_subs.empty():
                topic = self._pending_data_subs.get()
                self._sub.setsockopt(self._zmq.SUBSCRIBE, topic.encode("utf-8"))
            events = dict(poller.poll(timeout=self._poll_timeout_ms))
            if self._sub in events:
                _topic_frame, raw = self._sub.recv_multipart()
                topic, tact, payload = codec.decode(raw)
                with self._callbacks_lock:
                    callbacks = list(self._data_callbacks.get(topic, ()))
                for cb in callbacks:
                    cb(topic, tact, payload)

    def _cmd_loop(self) -> None:
        poller = self._zmq.Poller()
        poller.register(self._pull, self._zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(timeout=self._poll_timeout_ms))
            if self._pull in events:
                raw = self._pull.recv()
                topic, tact, payload = codec.decode(raw)
                with self._callbacks_lock:
                    callbacks = list(self._cmd_callbacks)
                for cb in callbacks:
                    cb(topic, tact, payload)

    def bound_data_endpoint(self) -> str | None:
        """Фактический адрес PUB-сокета (нужен, когда `data_bind` содержит `*`/порт 0)."""
        if self._pub is None:
            return None
        return self._pub.getsockopt_string(self._zmq.LAST_ENDPOINT)

    def bound_cmd_endpoint(self) -> str | None:
        """Фактический адрес PULL-сокета (см. `bound_data_endpoint`)."""
        if self._pull is None:
            return None
        return self._pull.getsockopt_string(self._zmq.LAST_ENDPOINT)

    def close(self) -> None:
        self._stop.set()
        for thread in (self._data_thread, self._cmd_thread):
            if thread is not None:
                thread.join(timeout=1.0)
        for sock in (self._pub, self._push, self._sub, self._pull):
            if sock is not None:
                sock.close(0)


class WebSocketTransport:
    """Publish-only WS-шлюз для браузера (raw ZMQ в браузере не работает, N4/4b).

    Держит `websockets`-сервер в СВОЁМ фоновом asyncio-цикле (отдельный тред);
    `publish()` из вызывающего (синхронного) кода планирует широковещательную
    рассылку УЖЕ msgpack-закодированных байт всем подключённым браузерам через
    `run_coroutine_threadsafe` -- та же схема `codec.py`, что и `ZmqTransport`
    (JS-декодер читает идентичный формат, см. `web/codec.js`).

    Обратный канал (браузер -> сервер, команды) в объём P6 НЕ входит (задача
    ограничивает веб-фронт визуализацией/контролами локально в браузере, команды
    "цели/помехи" панели -- десктоп через `ZmqTransport`) -- `subscribe()` кидает
    `NotImplementedError` явно, а не тихо no-op.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_connect: Callable[[], Iterable[tuple[str, int, object]]] | None = None,
    ) -> None:
        import asyncio

        import websockets

        self._asyncio: Any = asyncio
        self._websockets: Any = websockets
        self._host = host
        self._port = port
        self._clients: set[Any] = set()
        self._loop: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._server: Any = None
        # §4.4: реплей позднему клиенту -- при подключении шлём ему эти сообщения
        # (напр. `PanelPublisher.replay_messages`: meta + весь лог сессии), см. `_handle_client`.
        self._on_connect = on_connect

    def set_on_connect(
        self, on_connect: Callable[[], Iterable[tuple[str, int, object]]] | None
    ) -> None:
        """Зарегистрировать реплей позднему клиенту (§4.4): при подключении шлём эти сообщения.

        Вызывается, напр., `PanelPublisher.start()` (duck-typing) -- поздний браузер получает
        `meta` + снапшот лога сразу, а не ждёт следующего такта. Снимок берётся на КАЖДОЕ
        подключение (актуальный на момент), поэтому передаём callable, а не готовый список.
        """
        self._on_connect = on_connect

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_loop(self) -> None:
        loop = self._asyncio.new_event_loop()
        self._loop = loop
        self._asyncio.set_event_loop(loop)

        async def _main() -> None:
            # `max_size=None` -- сырой куб (nx,ny,N complex64) легко превышает
            # дефолтный лимит websockets (1 МБ); кадры сцены крупнее -- фрейминг
            # НЕ режем вручную (риск рассинхрона msgpack-границ), поднимаем лимит.
            self._server = await self._websockets.serve(
                self._handle_client, self._host, self._port, max_size=None
            )
            self._ready.set()
            await self._server.wait_closed()

        loop.run_until_complete(_main())

    async def _handle_client(self, websocket: Any) -> None:
        self._clients.add(websocket)
        try:
            if self._on_connect is not None:
                # §4.4: поздний клиент получает снапшот сессии (meta + весь лог) СРАЗУ при
                # подключении -- иначе увидел бы сцену только со следующего такта. Реплей идёт
                # ТОЛЬКО этому сокету (не broadcast). Callable вызывается на КАЖДОЕ подключение.
                for topic, tact, payload in self._on_connect():
                    await websocket.send(codec.encode(topic, tact, payload))
            async for _msg in websocket:
                pass   # входящие сообщения браузера игнорируются -- см. докстринг класса
        finally:
            self._clients.discard(websocket)

    def publish(self, topic: str, tact: int, payload: object) -> None:
        if self._thread is None:
            self.start()
        raw = codec.encode(topic, tact, payload)
        if self._loop is None or not self._clients:
            return
        self._asyncio.run_coroutine_threadsafe(self._broadcast(raw), self._loop)

    async def _broadcast(self, raw: bytes) -> None:
        stale = []
        for client in list(self._clients):
            try:
                await client.send(raw)
            except Exception:  # noqa: BLE001 -- отвалившийся клиент не должен ронять broadcast
                stale.append(client)
        for client in stale:
            self._clients.discard(client)

    def subscribe(self, topic: str, callback: Callback) -> None:
        raise NotImplementedError(
            "WebSocketTransport -- publish-only шлюз для браузера (P6), обратный канал "
            "(браузер -> сервер) не реализован -- команды идут через ZmqTransport десктоп-панели"
        )

    def close(self) -> None:
        if self._loop is not None and self._server is not None:
            self._asyncio.run_coroutine_threadsafe(self._close_server(), self._loop)
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    async def _close_server(self) -> None:
        self._server.close()


class FanOutTransport:
    """Composite: `publish` -- ОДИН вызов форвардится в КАЖДЫЙ вложенный транспорт (N4).

    "Один движок, два фронта": продюсер (`SceneServer`) не знает, сколько
    фронтов подписано -- вызывает `publish` один раз, `FanOutTransport` решает,
    куда доставить. `subscribe` форвардится в те вложенные транспорты, что его
    поддерживают (например, `WebSocketTransport.subscribe` не поддерживается --
    пропускается, а не роняет весь fan-out).
    """

    def __init__(self, transports: Sequence[Transport]) -> None:
        if not transports:
            raise ValueError("FanOutTransport: нужен хотя бы один вложенный транспорт")
        self._transports = list(transports)

    def publish(self, topic: str, tact: int, payload: object) -> None:
        for transport in self._transports:
            transport.publish(topic, tact, payload)

    def subscribe(self, topic: str, callback: Callback) -> None:
        supported = False
        errors: list[str] = []
        for transport in self._transports:
            try:
                transport.subscribe(topic, callback)
                supported = True
            except NotImplementedError as exc:
                errors.append(str(exc))
        if not supported:
            raise NotImplementedError(
                f"FanOutTransport: ни один вложенный транспорт не поддерживает "
                f"subscribe({topic!r}): {errors}"
            )

    def close(self) -> None:
        for transport in self._transports:
            close = getattr(transport, "close", None)
            if callable(close):
                close()
