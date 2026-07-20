"""core.runtime -- межпроцессный транспорт панели (P6, ZMQ/WebSocket + msgpack-кодек).

Отдельно от `core.data_context.MessageBus` (P1, ВНУТРИпроцессный Observer, A7):
здесь -- граница МЕЖДУ процессами (продюсер сцены <-> панель), см. `transport.py`
докстринг узла N3. Публичный набор: `codec` (кодек), `Transport`/`ZmqTransport`/
`WebSocketTransport`/`FanOutTransport` (транспорт), `SceneServer`/`SceneState`
(издатель такта + приём команд), `commands` (Command pattern панель->сервер).
"""
from __future__ import annotations

from . import codec
from .commands import (
    COMMAND_REGISTRY,
    AddTarget,
    Command,
    EnableJammer,
    RemoveTarget,
    SetMotion,
    SetNeighborPlanes,
    Step,
    decode_command,
)
from .panel_publisher import PanelPublisher, Tick, TickLog
from .raw_queue import RawFrame, RawQueue
from .raw_source import FileSource, RawCubeSource
from .scene_server import CMD_TOPIC, LiveTarget, SceneServer, SceneState
from .transport import FanOutTransport, Transport, WebSocketTransport, ZmqTransport

__all__ = [
    "codec",
    "Transport", "ZmqTransport", "WebSocketTransport", "FanOutTransport",
    "SceneServer", "SceneState", "LiveTarget", "CMD_TOPIC",
    "Command", "AddTarget", "RemoveTarget", "SetMotion", "EnableJammer",
    "Step", "SetNeighborPlanes", "decode_command", "COMMAND_REGISTRY",
    "Tick", "TickLog", "PanelPublisher",
    "RawFrame", "RawQueue", "RawCubeSource", "FileSource",
]
