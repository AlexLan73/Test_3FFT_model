"""ex4-сервер — стример живой панели полёта ЧЕРЕЗ СОКЕТ (P6, канон §1.6).

Правильная архитектура (правило `.claude/rules/07-math-in-core.md` + handoff
`sessions/2026-07-19_HANDOFF_socket_panel.md`): математика/модели — в `core`,
панель — тонкий клиент. Сервер публикует такты сцены через `WebSocketTransport`
(msgpack, `core.runtime.codec` — язык-нейтрально), браузер (`web/`) их рисует.

Отличие от прошлой самодостаточной страницы (в архиве `архив/ex4_web_selfcontained_*.zip`):
там вся история 30 тактов вшивалась статически в HTML и проекция портировалась в JS.
Здесь сервер — ИСТОЧНИК данных: гоняет `Ex4Flight.run_history` (R5 — один прогон)
ОДИН раз, затем СТРИМИТ такты по сокету. Клиент не считает сцену — только рисует
пришедшие МИРОВЫЕ координаты, проецируя единой камерой `core.graphics.Projection`
(параметры `Projection.as_js()` — тот же источник знаков осей, что тесты `test_camera.py`).

Каналы (topic):
  "meta" — метаданные сессии (nx/ny/оси/камера/станции/финальные признаки), шлётся
           в начале каждого прохода истории (поздний клиент получит в пределах цикла);
  "tick" — один такт: истина, точки-детекции, треки со следом, срезы (кроп ±8).

Запуск:  .venv/Scripts/python.exe demo/ex4_flight/server.py            # канон 30 тактов, порт 8765
         .venv/Scripts/python.exe demo/ex4_flight/server.py --tacts 6 --port 8765
Затем открыть web/index.html в браузере и нажать «Подключиться».
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Конвенция репо: работает форма `python demo/ex4_flight/server.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from core.graphics import Projection  # noqa: E402  — ЕДИНАЯ модель камеры (математика в core)
from core.runtime import PanelPublisher, TickLog, WebSocketTransport  # noqa: E402
from demo.ex4_flight.example import Ex4Flight, Ex4Params, TactRecord  # noqa: E402

CROP_HALF = 8    # окно среза ±8 бинов вокруг трека (решение Alex 2A)

# Стационарные помехи-станции (не двигаются, «работают» с фиксированного угла), каждая
# со своим значком. Размещены в свободных углах поля, вне полосы barrage и трасс.
STATIONS = (
    {"type": "radar", "label": "другой радар (CW)", "kx": -22.0, "ky": -21.0},
    {"type": "ham", "label": "радиолюбитель", "kx": 21.0, "ky": -24.0},
)


def _crop_slice(sl: dict[str, Any], nx: int, ny: int, half: int) -> dict[str, Any]:
    """Кроп energy_db вокруг углового бина трека: {x0, y0, m[(2h+1)×(2h+1)]}."""
    e = sl["energy_db"]
    w = 2 * half + 1
    ix = int(round(sl["kx"])) + nx // 2
    iy = int(round(sl["ky"])) + ny // 2
    x0 = int(np.clip(ix - half, 0, max(0, nx - w)))
    y0 = int(np.clip(iy - half, 0, max(0, ny - w)))
    m = e[x0:x0 + w, y0:y0 + w]
    return {"x0": x0, "y0": y0, "m": [[round(float(v), 1) for v in row] for row in m]}


def tick_payload(rec: TactRecord, p: Ex4Params, k_trail: int) -> dict[str, Any]:
    """Один такт → примитивы (МИРОВЫЕ координаты; проекция — на клиенте единой камерой)."""
    tgt, comb = rec.truth["target"], rec.truth["comb"]
    bkx, bky, _ = rec.truth["barrage"]
    return {
        "truth": {"t": [round(tgt[0], 2), round(tgt[1], 2), int(tgt[2])],
                  "c": [round(comb[0], 2), round(comb[1], 2), int(comb[2])],
                  "b": [round(bkx, 2), round(bky, 2)]},
        "band": [round(v, 1) for v in rec.band_angle] if rec.banded else None,
        "pts": [[round(q.kx, 1), round(q.ky, 1), int(q.pos), round(q.db, 1)]
                for q in rec.points],
        "trk": [{"id": t["id"], "kx": round(t["kx"], 2), "ky": round(t["ky"], 2),
                 "mv": int(t["is_moving"]),
                 "h": [[round(h[1], 2), round(h[2], 2)] for h in t["history"][-k_trail:]]}
                for t in rec.tracks],
        "sl": [{"id": s["track_id"], "kx": round(float(s["kx"]), 1),
                "ky": round(float(s["ky"]), 1), "pos": int(s["pos"]),
                "mv": int(s["is_moving"]),
                **_crop_slice(s, p.nx, p.ny, CROP_HALF)}
               for s in rec.slices],
    }


def meta_payload(ex: Ex4Flight) -> dict[str, Any]:
    """Метаданные сессии (единожды на цикл): апертура, камера, станции, финал-признаки."""
    p = ex._p
    final_feats: dict[str, Any] = {}
    if ex._history:
        for s in ex._history[-1].slices:
            if s["features"]:
                final_feats[str(s["track_id"])] = {
                    "label": s["label"],
                    "f": {k: round(float(v), 4) for k, v in s["features"].items()},
                }
    return {
        "nx": p.nx, "ny": p.ny, "nAxis": p.n_axis, "kTrail": p.k_trail,
        "nTicks": len(ex._history),
        "stats": {k: str(v) for k, v in ex._stats.items()},
        "finalFeats": final_feats,
        "stations": [dict(s) for s in STATIONS],
        "cam": Projection(nx=p.nx, ny=p.ny, n_range=p.n_axis).as_js(),
    }


def build_session(ex: Ex4Flight) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Прогнать историю один раз → (meta, [tick, ...]) готовые к публикации примитивы."""
    p = ex._p
    ticks = [tick_payload(rec, p, p.k_trail) for rec in ex._history]
    return meta_payload(ex), ticks


def serve(port: int, tacts: int | None, delay_s: float, cycles: int | None) -> None:
    """Прогнать ex4 один раз, поднять WS-шлюз, СТРИМить такты по кругу (живая панель).

    Источник (предрасчёт `run_history`) развязан с транспортом через `PanelPublisher`
    (Этап A спеки realtime_panel) — тот же публикатор, что `live_demo.py`. `TickLog`
    с `cap=len(ticks)` держит ровно один проход истории (бесконечный реплей не растит
    память); `republish_meta()` в начале каждого прохода — поздний клиент (slow joiner, N1).
    """
    params = Ex4Params(tacts=90) if tacts is None else Ex4Params(tacts=tacts)
    ex = Ex4Flight(params=params)
    ex.run_history(np.random.default_rng(ex.seed))
    meta, ticks = build_session(ex)

    publisher = PanelPublisher(WebSocketTransport(port=port), log=TickLog(cap=len(ticks)))
    publisher.start()
    print(f"ex4-server: ws://127.0.0.1:{port} · {meta['nTicks']} тактов · {ex._stats}")
    print("  открыть web/index.html в браузере → «Подключиться». Ctrl+C — стоп.")

    done = 0
    try:
        while cycles is None or done < cycles:
            # meta в начале КАЖДОГО прохода: поздно подключившийся клиент получит
            # апертуру/камеру/станции в пределах одного цикла истории (slow joiner, N1).
            publisher.push_meta(meta) if done == 0 else publisher.republish_meta()
            for i, tk in enumerate(ticks):
                publisher.push_tick(i, tk)
                time.sleep(delay_s)
            done += 1
    except KeyboardInterrupt:
        print("\nex4-server: остановлен.")
    finally:
        publisher.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Сокет-сервер живой панели ex4 (канон §1.6).")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket-порт (дефолт 8765)")
    parser.add_argument("--tacts", type=int, default=None, help="число тактов (дефолт 90)")
    parser.add_argument("--delay", type=float, default=0.2, help="пауза между тактами, с (дефолт 0.2)")
    parser.add_argument("--cycles", type=int, default=None,
                        help="сколько раз прогнать историю по кругу (дефолт — бесконечно)")
    args = parser.parse_args()
    serve(args.port, args.tacts, args.delay, args.cycles)


if __name__ == "__main__":
    main()
