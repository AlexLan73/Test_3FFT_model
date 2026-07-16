"""codec -- ЕДИНСТВЕННОЕ место (де)кодирования msgpack-сообщений транспорта (P6, N2).

🚫 НЕ `pickle` (py-специфично, C++/JS не разберут). 🚫 НЕ `msgpack-numpy` (это тоже
py-специфичное РАСШИРЕНИЕ msgpack -- ext-типы понимает только та же библиотека на
питоне). Комплексные/вещественные массивы кодируются ЯВНОЙ схемой поверх обычных
примитивов msgpack (map/str/int/bin) -- любой язык с msgpack-библиотекой (C++,
JS/`@msgpack/msgpack`, ...) читает её без спец-плагинов.

## Схема сообщения (msgpack map, ключи -- ASCII-строки)

```
{
  "topic": str,   # имя канала: "cube" | "squares" | "tracks" | "cmd" | ...
  "tact":  int,   # номер такта (>=0); у команд панель->сервер обычно 0 (не используется)
  "kind":  str,   # "array" -- закодирован numpy-массив, "value" -- произвольные примитивы
  "payload": ...  # см. ниже, форма зависит от "kind"
}
```

`kind == "array"` (каналы `cube`, `squares`) -- `payload`:
```
{
  "shape": [int, ...],   # форма массива, C-порядок (row-major)
  "dtype": str,           # "complex64" | "complex128" | "float32" | "float64" |
                          # "int32" | "int64" | "uint8"
  "data":  bytes          # сырые байты, ВСЕГДА little-endian, ровно
                          # prod(shape) * itemsize(dtype) байт, C-порядок (`.tobytes()`)
}
```
Комплексные dtype (`complex64`=2x float32, `complex128`=2x float64) -- в msgpack
НЕТ родного complex-типа, поэтому по паре (re, im) на элемент, ЗАПИСАННЫХ ПОДРЯД
(interleaved), как в родном представлении numpy `complex64`/`complex128` -- то
есть `data` -- это ПРОСТО `arr.astype('<c8').tobytes()` (little-endian complex64),
С++/JS читает так же, как `numpy` пишет: `[re0, im0, re1, im1, ...]` float32.

`kind == "value"` (каналы `tracks`, `cmd`, ...) -- `payload` -- произвольная
УЖЕ-примитивная (str/int/float/bool/None/list/dict) вложенная структура. Кодек НЕ
умеет сериализовать numpy-массивы/dataclass'ы внутри `value` -- вызывающий код
(`scene_server.py`) обязан привести их к примитивам ДО `encode()` (см.
`SceneServer._targets_payload`), иначе `msgpack.packb` кинет `TypeError`.

## Почему единственное место

Схема -- контракт для будущего C++/GPU-продюсера (P7) и JS-панели (`web/`, тот же
`decode()` переписан на JS в `web/codec.js`, ключи/формы совпадают 1:1). Если
кодирование/декодирование появится где-то ещё -- схема неизбежно разъедется.
"""
from __future__ import annotations

from typing import Any

import msgpack
import numpy as np

# dtype-имя (строка схемы) <-> numpy dtype. Явный little-endian ('<') -- фикс
# endianness (N2): хост может быть big-endian, на проводе -- ВСЕГДА LE.
_DTYPE_TO_NAME: dict[np.dtype, str] = {
    np.dtype("<c8"): "complex64",
    np.dtype("<c16"): "complex128",
    np.dtype("<f4"): "float32",
    np.dtype("<f8"): "float64",
    np.dtype("<i4"): "int32",
    np.dtype("<i8"): "int64",
    np.dtype("<u1"): "uint8",
}
_NAME_TO_DTYPE: dict[str, np.dtype] = {name: dt for dt, name in _DTYPE_TO_NAME.items()}


def _dtype_name(dtype: np.dtype) -> str:
    le_dtype = np.dtype(dtype).newbyteorder("<")
    name = _DTYPE_TO_NAME.get(le_dtype)
    if name is None:
        raise ValueError(
            f"codec: неподдерживаемый dtype={dtype} -- разрешены {sorted(_NAME_TO_DTYPE)}"
        )
    return name


def encode(topic: str, tact: int, payload: object) -> bytes:
    """`(topic, tact, payload)` -> msgpack-байты (единственная точка кодирования, N2).

    `payload` -- либо `np.ndarray` (кодируется схемой "array"), либо уже-примитивная
    структура (кодируется схемой "value" как есть).
    """
    if isinstance(payload, np.ndarray):
        arr = np.ascontiguousarray(payload).astype(
            np.dtype(payload.dtype).newbyteorder("<"), copy=False
        )
        body: dict[str, Any] = {
            "topic": topic,
            "tact": int(tact),
            "kind": "array",
            "payload": {
                "shape": list(arr.shape),
                "dtype": _dtype_name(arr.dtype),
                "data": arr.tobytes(order="C"),
            },
        }
    else:
        body = {"topic": topic, "tact": int(tact), "kind": "value", "payload": payload}
    return msgpack.packb(body, use_bin_type=True)


def decode(raw: bytes) -> tuple[str, int, object]:
    """msgpack-байты -> `(topic, tact, payload)`, обратное к `encode` (единственная точка декода).

    Для `kind == "array"` восстанавливает `np.ndarray` (копия -- `frombuffer` над
    временным `bytes`, владение памятью не шарим наружу). Для `kind == "value"`
    возвращает `payload` как есть (примитивы msgpack -> str/int/float/bool/None/list/dict).
    """
    body = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    topic = body["topic"]
    tact = body["tact"]
    kind = body["kind"]
    payload = body["payload"]
    if kind == "array":
        dtype = _NAME_TO_DTYPE.get(payload["dtype"])
        if dtype is None:
            raise ValueError(f"codec: неизвестный dtype в сообщении: {payload['dtype']!r}")
        shape = tuple(payload["shape"])
        arr = np.frombuffer(payload["data"], dtype=dtype).reshape(shape).copy()
        return topic, tact, arr
    if kind == "value":
        return topic, tact, payload
    raise ValueError(f"codec: неизвестный kind={kind!r} (ожидался 'array' или 'value')")


def encode_command(cmd: str, args: dict[str, Any]) -> bytes:
    """Команда панель->сервер: та же схема "value" под фиксированным topic="cmd".

    `payload = {"cmd": cmd, "args": args}` -- `args` обязаны быть примитивами
    (см. `commands.py::Command.to_message`).
    """
    return encode("cmd", 0, {"cmd": cmd, "args": args})


def decode_command(raw: bytes) -> tuple[str, dict[str, Any]]:
    """Обратное к `encode_command` -- `(cmd_name, args)`."""
    topic, _tact, payload = decode(raw)
    if topic != "cmd" or not isinstance(payload, dict) or "cmd" not in payload:
        raise ValueError(f"codec: сообщение не похоже на команду: topic={topic!r}, payload={payload!r}")
    return str(payload["cmd"]), dict(payload.get("args") or {})
