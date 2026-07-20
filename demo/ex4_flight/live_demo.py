"""live_demo -- живой демо-источник панели ex4 (Этап A спеки realtime_panel, БЕЗ GPU/куба).

Отличие от `server.py` (предрасчёт `Ex4Flight.run_history` → стрим по кругу): здесь ИСТОЧНИК
и ТРАНСПОРТ развязаны через `PanelPublisher` (`core/runtime/panel_publisher.py`) -- реалтайм-
цикл формирует такт ПРЯМО СЕЙЧАС (детерминированная летящая точка-заглушка, без куба/GPU) и
публикует его с реальной паузой `time.sleep`, доказывая реалтайм end-to-end (§3 Этап A спеки
`MemoryBank/specs/realtime_panel_2026-07-19.md`).

`build_live_tick(step, cfg)` -- ЧИСТАЯ функция (без сети/sleep, тестируема): такт §2.1
целиком определяется номером шага `step` (детерминизм). Реалтайм-цикл (`serve`) лишь
дёргает её по кругу и публикует результат.

Запуск:  .venv/bin/python demo/ex4_flight/live_demo.py            # бесконечный поток, порт 8765
         .venv/bin/python demo/ex4_flight/live_demo.py --ticks 50 --delay 0.1
Затем открыть web/index.html в браузере и нажать «Подключиться».
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Конвенция репо: работает форма `python demo/ex4_flight/live_demo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from core.graphics import Projection  # noqa: E402  -- ЕДИНАЯ модель камеры (математика в core)
from core.runtime import PanelPublisher, WebSocketTransport  # noqa: E402
from demo.ex4_flight.server import STATIONS  # noqa: E402  -- переиспользуем, не копируем

_REPUBLISH_EVERY = 30   # каждые N тактов повторно шлём meta -- поздний клиент (WS publish-only)


@dataclass(frozen=True, slots=True)
class LiveSceneConfig:
    """Параметры детерминированной летящей точки-заглушки (Value Object, Этап A).

    Апертура/дальность/след -- те же по умолчанию, что и `demo/ex4_flight/server.py`
    (64×64×4096, `kTrail=8`), чтобы панель конфигурировать не пришлось.
    """

    nx: int = 64
    ny: int = 64
    n_axis: int = 4096
    k_trail: int = 8
    period: int = 120     # тактов на один "перелёт" из угла в угол (пинг-понг, без разрыва)
    seed: int = 7          # база детерминированного rng слабых точек


def _target_position(step: int, cfg: LiveSceneConfig) -> tuple[float, float, int]:
    """Позиция цели на шаге `step` -- детерминированный пинг-понг по диагонали поля.

    Без разрывов (в отличие от развёртки с обнулением): дошла до угла -- летит назад.
    """
    margin = 4.0
    half_x = cfg.nx / 2 - margin
    half_y = cfg.ny / 2 - margin
    phase = step % (2 * cfg.period)
    frac = phase / cfg.period if phase <= cfg.period else 2.0 - phase / cfg.period
    kx = -half_x + 2 * half_x * frac
    ky = -half_y + 2 * half_y * frac
    r_min, r_max = cfg.n_axis // 4, cfg.n_axis // 2
    r = int(round(r_min + (r_max - r_min) * frac))
    return round(kx, 2), round(ky, 2), r


def build_live_tick(step: int, cfg: LiveSceneConfig) -> dict[str, Any]:
    """Один такт живого потока (контракт §2.1) -- ЧИСТАЯ функция от `step` (без сети/sleep).

    `truth.c` (носитель) и `truth.b` (barrage) зафиксированы в углах поля -- панель терпит
    неподвижную "истину" второстепенных объектов, "живёт" только цель `truth.t` + её трек.
    """
    kx, ky, r = _target_position(step, cfg)
    margin = 4.0
    half_x = cfg.nx / 2 - margin
    half_y = cfg.ny / 2 - margin
    r_min = cfg.n_axis // 4

    rng = np.random.default_rng(cfg.seed + step)   # seeded -- детерминизм (запрет random/time в логике)
    weak_pts = [
        [round(float(rng.uniform(-half_x, half_x)), 1),
         round(float(rng.uniform(-half_y, half_y)), 1),
         int(rng.integers(0, cfg.n_axis)),
         round(float(rng.uniform(5.0, 15.0)), 1)]
        for _ in range(3)
    ]

    history_from = max(0, step - cfg.k_trail + 1)
    history = [[hx, hy] for hx, hy, _hr in
               (_target_position(s, cfg) for s in range(history_from, step + 1))]

    return {
        "truth": {
            "t": [kx, ky, r],
            "c": [round(-half_x, 2), round(-half_y, 2), r_min],
            "b": [round(half_x * 0.5, 2), round(-half_y * 0.5, 2)],
        },
        "band": None,
        "pts": [[kx, ky, r, 30.0], *weak_pts],
        "trk": [{"id": 1, "kx": kx, "ky": ky, "mv": 1, "h": history}],
        "sl": [],
        "feats": None,
    }


def build_live_meta(cfg: LiveSceneConfig, n_ticks: int | None) -> dict[str, Any]:
    """Метаданные сессии (единожды на старт) -- апертура/камера/станции (контракт §2.1)."""
    return {
        "nx": cfg.nx, "ny": cfg.ny, "nAxis": cfg.n_axis, "kTrail": cfg.k_trail,
        "nTicks": n_ticks,
        "stats": {},
        "finalFeats": {},
        "stations": [dict(s) for s in STATIONS],
        "cam": Projection(nx=cfg.nx, ny=cfg.ny, n_range=cfg.n_axis).as_js(),
    }


def serve(port: int, ticks: int | None, delay_s: float) -> None:
    """Поднять WS-шлюз и реалтайм-цикл: формировать такт ПРЯМО СЕЙЧАС и публиковать с паузой."""
    cfg = LiveSceneConfig()
    transport = WebSocketTransport(port=port)
    publisher = PanelPublisher(transport)
    publisher.start()

    meta = build_live_meta(cfg, ticks)
    publisher.push_meta(meta)
    print(f"live_demo: ws://127.0.0.1:{port} · nTicks={meta['nTicks']} (None=бесконечно)")
    print("  открыть web/index.html в браузере → «Подключиться». Ctrl+C — стоп.")

    step = 0
    try:
        while ticks is None or step < ticks:
            publisher.push_tick(step, build_live_tick(step, cfg))
            if step > 0 and step % _REPUBLISH_EVERY == 0:
                publisher.republish_meta()   # поздний клиент -- WS publish-only, без on_connect
            time.sleep(delay_s)
            step += 1
    except KeyboardInterrupt:
        print("\nlive_demo: остановлен.")
    finally:
        publisher.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Живой демо-источник панели ex4 (Этап A realtime_panel, без GPU/куба)."
    )
    parser.add_argument("--port", type=int, default=8765, help="WebSocket-порт (дефолт 8765)")
    parser.add_argument("--ticks", type=int, default=None, help="число тактов (дефолт — бесконечно)")
    parser.add_argument("--delay", type=float, default=0.2, help="пауза между тактами, с (дефолт 0.2)")
    args = parser.parse_args()
    serve(args.port, args.ticks, args.delay)


if __name__ == "__main__":
    main()
