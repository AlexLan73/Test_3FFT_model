"""Демо P6 body-motion: `SceneServer` (продюсер) + живая сокет-панель (Composition Root).

Запускает `SceneServer` на локальном ZMQ (данные -- PUB/SUB, команды -- PUSH/PULL,
см. `core/runtime/transport.py`) + опционально WS-шлюз для `web/` дашборда через
`FanOutTransport` ("один движок, два фронта", N4) + опционально Dear PyGui панель
в этом же процессе (десктоп-фронт, N5 -- если `dearpygui` не установлен/нет дисплея,
падает в headless-режим: сервер крутится, печатает такты в консоль).

Запуск:
    python demo_body_motion_panel.py                    # headless: N_TACTS тактов, лог в консоль
    python demo_body_motion_panel.py --panel             # + Dear PyGui окно (нужен дисплей)
    python demo_body_motion_panel.py --web-port 8765     # + WebSocket-шлюз для web/index.html
    python demo_body_motion_panel.py --snapshot-html out/data/p6_snapshot.html   # + plotly-снимок
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from core.config import JammerFlags, ProjectConfig, SceneConfig
from core.runtime.commands import AddTarget
from core.runtime.scene_server import SceneServer, SceneState
from core.runtime.transport import FanOutTransport, WebSocketTransport, ZmqTransport

N_TACTS = 40
DT = 1.0


def _initial_state(seed: int | None) -> SceneState:
    """3 старта (реюз идеи `demo_body_motion_multi.py`) через `AddTarget.apply` --
    не дублируем логику построения `LiveTarget`, команда и демо строят цель одинаково.
    `motion` -- ИМЯ (не python-класс, N2: то же самое пошло бы по проводу от живой
    панели), см. `commands._build_motion` за соответствием имя -> `MotionModel`."""
    state = SceneState()
    rng = np.random.default_rng(seed)
    bands = (
        ("cv", (1800.0, 600.0, -8000.0), (0.0, 0.0, 130.0)),
        ("markov", (-2200.0, -900.0, -8500.0), (0.0, 0.0, 140.0)),
        ("turn", (0.0, 2200.0, -7500.0), (0.0, 0.0, 120.0)),
    )
    for motion, pos, vel in bands:
        AddTarget(pos=pos, vel=vel, motion=motion,
                  seed=int(rng.integers(0, 2**31 - 1))).apply(state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="P6 body-motion panel demo (SceneServer + панель).")
    parser.add_argument("--tacts", type=int, default=N_TACTS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--panel", action="store_true", help="запустить Dear PyGui окно (нужен дисплей)")
    parser.add_argument("--web-port", type=int, default=None,
                        help="поднять WebSocketTransport на этом порту для web/index.html")
    parser.add_argument("--data-port", type=int, default=0, help="0 = ZMQ сам выберет свободный порт")
    parser.add_argument("--cmd-port", type=int, default=0)
    parser.add_argument("--snapshot-html", type=str, default=None,
                        help="путь -- сохранить plotly-снимок последнего такта (нужен extras `viz`)")
    args = parser.parse_args()

    cfg = ProjectConfig(scene=SceneConfig(jammers=JammerFlags(barrage=False)))

    zmq_transport = ZmqTransport(
        data_bind=f"tcp://127.0.0.1:{args.data_port}",
        cmd_bind=f"tcp://127.0.0.1:{args.cmd_port}",
    )
    print(f"ZMQ данные: {zmq_transport.bound_data_endpoint()}")
    print(f"ZMQ команды: {zmq_transport.bound_cmd_endpoint()}")

    transport = zmq_transport
    ws_transport: WebSocketTransport | None = None
    if args.web_port is not None:
        ws_transport = WebSocketTransport(port=args.web_port)
        ws_transport.start()
        transport = FanOutTransport([zmq_transport, ws_transport])
        print(f"WebSocket шлюз для web/index.html: ws://127.0.0.1:{args.web_port}")

    state = _initial_state(args.seed)
    server = SceneServer(cfg, transport, state, seed=args.seed)

    last_vol: np.ndarray | None = None
    if args.panel:
        from core.graphics.panel.panel_app import PanelApp

        panel_transport = ZmqTransport(
            data_connect=zmq_transport.bound_data_endpoint(),
            cmd_connect=zmq_transport.bound_cmd_endpoint(),
        )
        panel = PanelApp(panel_transport, cfg)

        stop_after = time.monotonic() + 1e9   # панель управляет остановкой сама (закрытие окна)

        def _pump_server() -> bool:
            result = server.step()
            if result is not None:
                nonlocal last_vol
                last_vol = result[1]
            time.sleep(0.05)
            return time.monotonic() > stop_after

        panel.run(should_stop=_pump_server)
    else:
        for i in range(args.tacts):
            result = server.step()
            if result is not None:
                last_vol = result[1]
                print(f"такт {i}: {len(server.state.targets)} целей, vol.shape={last_vol.shape}")
            time.sleep(0.02)

    if args.snapshot_html and last_vol is not None:
        try:
            from core.generators.waveforms import LfmToCube
            from core.graphics.interactive import HtmlWriter, InteractiveCubeVisualizer

            cube = LfmToCube().fill(last_vol, cfg)
            fig = InteractiveCubeVisualizer().render(cube)
            out_dir, name = args.snapshot_html.rsplit("/", 1) if "/" in args.snapshot_html \
                else (".", args.snapshot_html)
            path = HtmlWriter(out_dir).write(fig, name)
            print(f"plotly-снимок записан: {path}")
        except ImportError:
            print("plotly не установлен (extras `viz`) -- снимок пропущен")

    zmq_transport.close()
    if ws_transport is not None:
        ws_transport.close()


if __name__ == "__main__":
    main()
