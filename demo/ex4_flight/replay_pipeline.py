"""replay_pipeline -- сквозное демо ВСЕЙ реалтайм-проводки БЕЗ GPU (§4.1/§7 спеки).

Доказывает, что вся проводка (диск -> `FileSource` -> `RawQueue` (producer-тред) ->
обработка (consumer) -> `PanelPublisher.push_tick` -> панель) работает end-to-end,
ДО того как появится реальная GPU-обработка. Позже GPU-чат заменит заглушку
`naive_cube_to_tick` на настоящий `SignalFrontend[sig]` + ядро детекции (Этап B,
`core/`) -- сама проводка (очередь/треды/publisher) при этом НЕ изменится.

⚠️ `naive_cube_to_tick` -- ЗАГЛУШКА ПРОВОДКИ, НЕ модель и НЕ DSP. Это placeholder:
argmax по модулю сырого куба -> одна точка-детекция. Реальная обработка (дечирп/
FFT/детекция) -- Этап B в `core`, делает GPU-чат. Здесь НЕ реализуется настоящий
`SignalFrontend`/детектор.

Запуск:  .venv/bin/python demo/ex4_flight/replay_pipeline.py                 # сгенерить набор во temp
         .venv/bin/python demo/ex4_flight/replay_pipeline.py --dir out/raw_demo --ticks 30 --delay 0.2
Затем открыть web/index.html в браузере и нажать «Подключиться».
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

# Конвенция репо: работает форма `python demo/ex4_flight/replay_pipeline.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from core.graphics import Projection  # noqa: E402  -- ЕДИНАЯ модель камеры (математика в core)
from core.runtime import (  # noqa: E402
    FileSource,
    PanelPublisher,
    RawFrame,
    RawQueue,
    WebSocketTransport,
)
from demo.ex4_flight.server import STATIONS  # noqa: E402  -- переиспользуем, не копируем


def _object_cell(step: int, nx: int, ny: int, n_axis: int) -> tuple[int, int, int]:
    """Детерминированная диагональная траектория «объекта» набора (пинг-понг по x/y).

    Тот же стиль, что `live_demo._target_position` -- без разрывов, шаг за шагом
    из угла в угол. Дальность растёт линейно с шагом (по модулю периода по оси r).
    """
    period = max(nx, ny)
    phase = step % (2 * period)
    frac = phase / period if phase <= period else 2.0 - phase / period
    margin = 2
    ix = int(round(margin + (nx - 1 - 2 * margin) * frac))
    iy = int(round(margin + (ny - 1 - 2 * margin) * frac))
    ir = int((step * 7) % n_axis)   # своя, менее тривиальная периодичность по дальности
    return ix, iy, ir


def write_demo_dataset(
    directory: str | Path,
    n_ticks: int = 30,
    nx: int = 16,
    ny: int = 16,
    n_axis: int = 64,
    seed: int = 7,
) -> list[Path]:
    """Сгенерировать мини-набор сырых кубов `tick_00000.npy`... в `directory` (детерминизм).

    Каждый файл -- один сырой комплексный куб `(nx,ny,n_axis)` complex64: слабый seeded-шум
    + ОДИН сильный детерминированный пик-«объект» в ячейке `_object_cell(step, ...)`
    (летит по диагонали, как `live_demo`). Возвращает отсортированный список путей набора.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths: list[Path] = []
    for step in range(n_ticks):
        noise = (rng.standard_normal((nx, ny, n_axis)) + 1j * rng.standard_normal((nx, ny, n_axis)))
        cube = (0.1 * noise).astype(np.complex64)
        ix, iy, ir = _object_cell(step, nx, ny, n_axis)
        cube[ix, iy, ir] = np.complex64(50.0 + 0.0j)   # сильный пик -- находимый argmax'ом
        path = directory / f"tick_{step:05d}.npy"
        np.save(path, cube)
        paths.append(path)
    return sorted(paths)


def naive_cube_to_tick(frame: RawFrame, nx: int, ny: int) -> dict[str, Any]:
    """ЗАГЛУШКА проводки (НЕ обработка!) -- argmax по модулю куба -> одна точка-детекция.

    Реальный `SignalFrontend[sig]` + ядро детекции -- Этап B (`core/`), делает GPU-чат.
    Здесь только тривиальный placeholder, чтобы доказать проводку сырьё->tick->панель.
    """
    mag = np.abs(frame.cube)
    flat_idx = int(np.argmax(mag))
    ix, iy, ir = np.unravel_index(flat_idx, mag.shape)
    ix, iy, ir = int(ix), int(iy), int(ir)
    peak = float(mag[ix, iy, ir])
    db = round(float(20.0 * np.log10(max(peak, 1e-12))), 1)

    kx = ix - nx // 2
    ky = iy - ny // 2
    pos = ir

    return {
        "truth": None,
        "band": None,
        "pts": [[kx, ky, pos, db]],
        "trk": [{"id": 1, "kx": kx, "ky": ky, "mv": 1, "h": [[kx, ky]]}],
        "sl": [],
        "feats": None,
    }


def build_pipeline_meta(nx: int, ny: int, n_axis: int, n_ticks: int | None, sig: str) -> dict[str, Any]:
    """Метаданные сессии (единожды на старт) -- апертура/камера/станции (контракт §2.1)."""
    return {
        "nx": nx, "ny": ny, "nAxis": n_axis, "kTrail": 1,
        "nTicks": n_ticks,
        "sig": sig,
        "stats": {},
        "finalFeats": {},
        "stations": [dict(s) for s in STATIONS],
        "cam": Projection(nx=nx, ny=ny, n_range=n_axis).as_js(),
    }


def process_dataset(
    directory: str | Path, sig: str = "lfm", nx: int = 16, ny: int = 16
) -> list[dict[str, Any]]:
    """ЧИСТЫЙ прогон набора (без сети/тредов) -- `FileSource.iter_frames()` -> заглушка -> tick'и."""
    source = FileSource(directory, sig=sig)
    return [naive_cube_to_tick(frame, nx, ny) for frame in source.iter_frames()]


def run_pipeline(
    directory: str | Path,
    sig: str = "lfm",
    port: int = 8765,
    delay_s: float = 0.2,
    nx: int = 16,
    ny: int = 16,
    n_axis: int = 64,
) -> None:
    """Сквозной прогон: producer-тред (`FileSource` -> `RawQueue`) + consumer (главный поток).

    Развязка producer<->consumer (§4.1): сырьё транзитом через `RawQueue`
    (consume-and-drop), панель получает только лёгкие обработанные `tick`.
    """
    directory = Path(directory)
    n_files = len(sorted(directory.glob("*.npy")))

    raw_queue = RawQueue(maxsize=4)
    source = FileSource(directory, sig=sig, delay_s=delay_s)
    stop = threading.Event()
    producer = threading.Thread(target=source.run, args=(raw_queue.put, stop), daemon=True)

    publisher = PanelPublisher(WebSocketTransport(port=port))
    publisher.start()
    meta = build_pipeline_meta(nx, ny, n_axis, n_files, sig)
    publisher.push_meta(meta)
    print(f"replay_pipeline: ws://127.0.0.1:{port} · nTicks={n_files} · sig={sig} · dir={directory}")
    print("  открыть web/index.html в браузере → «Подключиться». Ctrl+C — стоп.")

    producer.start()
    try:
        received = 0
        while received < n_files:
            frame = raw_queue.get(timeout=max(delay_s * 5, 1.0))
            if frame is None:
                if not producer.is_alive():
                    break
                continue
            publisher.push_tick(frame.index, naive_cube_to_tick(frame, nx, ny))
            received += 1
    except KeyboardInterrupt:
        print("\nreplay_pipeline: остановлен.")
    finally:
        stop.set()
        producer.join(timeout=5.0)
        publisher.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сквозное демо реалтайм-проводки БЕЗ GPU (заглушка-обработка, Этап A/D)."
    )
    parser.add_argument("--dir", type=str, default=None,
                         help="папка с набором *.npy (дефолт -- сгенерить во временную)")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket-порт (дефолт 8765)")
    parser.add_argument("--delay", type=float, default=0.2, help="пауза между тактами, с (дефолт 0.2)")
    parser.add_argument("--sig", type=str, default="lfm", help="тип сигнала набора (дефолт lfm)")
    parser.add_argument("--ticks", type=int, default=30, help="число тактов при генерации (дефолт 30)")
    args = parser.parse_args()

    if args.dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="replay_pipeline_")
        write_demo_dataset(tmp_dir, n_ticks=args.ticks)
        directory = tmp_dir
        print(f"replay_pipeline: сгенерирован временный набор -> {directory}")
    else:
        directory = args.dir
        if not any(Path(directory).glob("*.npy")):
            write_demo_dataset(directory, n_ticks=args.ticks)
            print(f"replay_pipeline: набор сгенерирован -> {directory}")

    run_pipeline(directory, sig=args.sig, port=args.port, delay_s=args.delay)


if __name__ == "__main__":
    main()
