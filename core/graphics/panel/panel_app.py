"""panel_app -- тонкая обвязка Dear PyGui поверх `PanelModel` (Observer, P6, N5).

🟡 N5: dearpygui НЕ ставится в headless-среде (нет дисплея) -- импорт под
`try/except`, `_HAS_DEARPYGUI=False` -> `PanelApp(...)` кидает `ImportError` с
понятным сообщением при попытке создать окно (а не при импорте МОДУЛЯ -- модуль
импортируется всегда, тесты решают САМИ, пропускать степ или нет, `SkipTest`).

Вся логика (закладка +-N, теплокарты, токены) -- в `panel_model.py` (GUI-free).
Этот файл только: (1) тред Transport.subscribe -> `queue.Queue` (N3: SUB-тред НЕ
трогает GUI напрямую), (2) `_drain()` каждый кадр вычитывает очередь и кормит
`PanelModel`, (3) рисует окна (квадрат 16x16 + контролы + лог) поверх готовых
структур модели.
"""
from __future__ import annotations

import queue
from collections.abc import Callable

import numpy as np

from ...config import ProjectConfig
from ...models.result import SpectralCube
from ...runtime.commands import AddTarget, Command, EnableJammer, SetNeighborPlanes, Step
from ...runtime.transport import CMD_TOPIC, Transport
from .panel_model import PanelModel

try:
    import dearpygui.dearpygui as dpg
    _HAS_DEARPYGUI = True
except ImportError:                    # pragma: no cover -- нет дисплея/библиотеки в CI
    dpg = None                          # type: ignore[assignment]
    _HAS_DEARPYGUI = False

_CMAP_COLD = np.array([13, 17, 23])     # #0d1117 -- фон (пусто)
_CMAP_HOT = np.array([248, 81, 73])     # #f85149 -- максимум (аналог "color_map" образца)


def _value_to_color(value: float) -> tuple[int, int, int, int]:
    """0..1 -> RGBA (лерп холодный/фон -> горячий/пик), лёгкий аналог `color_map.py` образца."""
    t = float(np.clip(value, 0.0, 1.0))
    rgb = (_CMAP_COLD * (1.0 - t) + _CMAP_HOT * t).astype(int)
    return int(rgb[0]), int(rgb[1]), int(rgb[2]), 255


class PanelApp:
    """Живая десктоп-панель (Dear PyGui): квадрат 16x16 + контролы + лог (Observer над `Transport`)."""

    def __init__(self, transport: Transport, cfg: ProjectConfig, model: PanelModel | None = None) -> None:
        if not _HAS_DEARPYGUI:
            raise ImportError(
                "dearpygui не установлен -- PanelApp недоступен (см. pyproject.toml [panel]); "
                "используйте PanelModel напрямую (GUI-free) или запускайте `web/` дашборд"
            )
        self._transport = transport
        self._cfg = cfg
        self._model = model or PanelModel(neighbor_planes=cfg.viz_neighbor_planes)
        self._queue: queue.Queue[tuple[str, int, object]] = queue.Queue()
        self._log: list[str] = []

        transport.subscribe("cube", self._on_frame)
        transport.subscribe("squares", self._on_frame)
        transport.subscribe("tracks", self._on_frame)

    # -- приём (N3: тред транспорта только кладёт в очередь, GUI не трогает) -----
    def _on_frame(self, topic: str, tact: int, payload: object) -> None:
        self._queue.put((topic, tact, payload))

    def _to_spectral_cube(self, volume: np.ndarray) -> SpectralCube:
        """Сырой объём (канал 'cube') -> `SpectralCube` для `PanelModel` (реюз `WaveformToCube`)."""
        from ...generators.waveforms import AmToCube, LfmToCube

        to_cube = LfmToCube() if self._cfg.modulation == "lfm" else AmToCube()
        return to_cube.fill(volume, self._cfg)

    def _drain(self) -> None:
        """Вычитать всё накопленное в очереди -> покормить `PanelModel` (вызывается раз в кадр)."""
        pending_tracks: tuple[int, object] | None = None
        pending_cube: tuple[int, object] | None = None
        while True:
            try:
                topic, tact, payload = self._queue.get_nowait()
            except queue.Empty:
                break
            if topic == "cube":
                pending_cube = (tact, payload)
            elif topic == "tracks":
                pending_tracks = (tact, payload)
            # "squares" -- сервер уже прислал свёрнутый вид; PanelModel пересчитывает
            # его сам из cube (`full_square`), отдельно не хранится (нет второго
            # источника истины -- дубли не копим).
        if pending_cube is not None:
            tact, raw_vol = pending_cube
            vol = np.asarray(raw_vol)   # payload -- object на границе Transport (см. Callback), сужаем тип
            self._model.ingest_cube(tact, self._to_spectral_cube(vol))
            self._log.append(f"такт {tact}: cube {vol.shape}")
        if pending_tracks is not None:
            tact, payload = pending_tracks
            if isinstance(payload, dict):
                self._model.ingest_tracks(tact, payload.get("targets", []), payload.get("jammers", []))

    # -- команды (панель -> сервер, PUSH/PULL) -----------------------------------
    def _send(self, command: Command) -> None:
        name, args = command.to_message()
        self._transport.publish(CMD_TOPIC, 0, {"cmd": name, "args": args})
        self._log.append(f"-> команда {name}({args})")

    def _cb_add_target(self, _sender: object, _data: object) -> None:
        pos = (float(np.random.uniform(-2000, 2000)), float(np.random.uniform(-1000, 1000)),
               float(-np.random.uniform(5000, 9000)))
        vel = (0.0, 0.0, float(np.random.uniform(90, 150)))
        self._send(AddTarget(pos=pos, vel=vel, motion="cv"))

    def _cb_toggle_barrage(self, _sender: object, checked: bool) -> None:
        self._send(EnableJammer(barrage=bool(checked)))

    def _cb_neighbor_planes(self, _sender: object, value: int) -> None:
        self._model.set_neighbor_planes(int(value))
        self._send(SetNeighborPlanes(n=int(value)))

    def _cb_step(self, _sender: object, _data: object) -> None:
        self._send(Step(dt=1.0))

    # -- построение окон -----------------------------------------------------------
    def _build_windows(self) -> None:
        assert dpg is not None
        with dpg.window(label="radar3d -- панель (P6)", tag="main_window"):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Добавить цель", callback=self._cb_add_target)
                dpg.add_checkbox(label="Заград", callback=self._cb_toggle_barrage)
                dpg.add_slider_int(label="+-N плоскостей", default_value=self._model.neighbor_planes,
                                   min_value=0, max_value=20, callback=self._cb_neighbor_planes)
                dpg.add_button(label="Шаг", callback=self._cb_step)
            with dpg.drawlist(width=320, height=320, tag="square_draw"):
                pass
            dpg.add_text("", tag="log_text")

    def _redraw(self) -> None:
        assert dpg is not None
        square = self._model.full_square()
        if square is not None:
            dpg.delete_item("square_draw", children_only=True)
            nx, ny = square.shape
            vmax = float(square.max()) or 1.0
            cell = 320 // max(nx, ny)
            for ix in range(nx):
                for iy in range(ny):
                    color = _value_to_color(float(square[ix, iy]) / vmax)
                    x0, y0 = ix * cell, iy * cell
                    dpg.draw_rectangle((x0, y0), (x0 + cell, y0 + cell), fill=color,
                                       parent="square_draw")
        dpg.set_value("log_text", "\n".join(self._log[-8:]))

    # -- запуск --------------------------------------------------------------------
    def run(self, should_stop: Callable[[], bool] | None = None) -> None:
        """Ручной render-loop (см. Dear PyGui docs `render-loop.rst`): `_drain` каждый кадр."""
        assert dpg is not None
        dpg.create_context()
        self._build_windows()
        dpg.create_viewport(title="radar3d -- панель P6", width=760, height=520)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running() and (should_stop is None or not should_stop()):
            self._drain()
            self._redraw()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()
