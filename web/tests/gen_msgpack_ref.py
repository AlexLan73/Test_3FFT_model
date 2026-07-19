"""Генератор эталона для web/tests/test_msgpack.mjs — кодирует схемой codec.py.

Запуск:  .venv/Scripts/python.exe web/tests/gen_msgpack_ref.py <dir>
Пишет <dir>/msgs.bin (кадры с 4-байтным BE-префиксом длины) + <dir>/ref.json (эталон decode).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.runtime import codec  # noqa: E402

MSGS = [
    ("meta", 0, {"nx": 64, "ny": 64, "nAxis": 4096,
                 "stations": [{"type": "radar", "kx": -22.0, "ky": -21.0}],
                 "cam": {"scene": {"az": 0.42, "el": -0.32, "fit": 0.7}, "az0": 0.42, "el0": -0.32},
                 "stats": {"target_found": "30/30"}, "empty": None, "flag": True}),
    ("tick", 5, {"truth": {"t": [1.5, -3.25, 2912], "b": [5.0, 18.0]}, "band": None,
                 "pts": [[1.0, 10.0, 2880, -3.4], [-7.0, -2.0, 3100, -15.2]],
                 "trk": [{"id": 10, "kx": 1.23, "ky": 10.0, "mv": 1, "h": [[1.1, 9.9], [1.2, 10.0]]}],
                 "sl": [{"id": 10, "kx": 1.0, "ky": 10.0, "pos": 2912, "mv": 1, "x0": 0, "y0": 2,
                         "m": [[-25.0, -3.1], [-12.4, -0.0]]}]}),
]


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    raws = [codec.encode(t, k, p) for (t, k, p) in MSGS]
    blob = b"".join(len(r).to_bytes(4, "big") + r for r in raws)
    (out_dir / "msgs.bin").write_bytes(blob)
    ref = [{"topic": t, "tact": k, "payload": p} for (t, k, p) in MSGS]
    (out_dir / "ref.json").write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
    print(f"эталон: {len(raws)} сообщений → {out_dir}")


if __name__ == "__main__":
    main()
