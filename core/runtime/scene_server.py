"""SceneServer -- издатель тактов сцены + приёмник команд панели (P6, Composition Root use).

Публикует `cube` (сырой комплексный объём `(nx,ny,N)`, тот же канал/семантика,
что `core.generators.volume.CUBE_CHANNEL`), `squares` (свёрнутый по дальности
квадрат `(nx,ny)`, `SquareView.reduce_square`, P5) и `tracks` (примитивы -- см.
`_targets_payload`/`_jammers_payload`) через инжектированный `Transport` (ZMQ/
FanOut) на каждый такт. Команды панель->сервер (см. `commands.py`) приходят
асинхронно из фонового треда транспорта, копятся в очереди и применяются к
`SceneState` В НАЧАЛЕ следующего `step()` (не мгновенно -- детерминированный
порядок: один такт = применить накопленные команды -> продвинуть цели -> собрать
объём -> опубликовать).

## Почему НЕ переиспользован `MultiTactSequence` для самого цикла тактов

`MultiTactSequence` (P4, `tact_sequence.py`) -- итератор для ФИКСИРОВАННОГО на
момент конструирования списка `TargetHandle` и ЗАРАНЕЕ известного `n_tacts`
(контракт класса, см. его докстринг M2/M4). `SceneServer` -- живой процесс, где
список целей меняется КОМАНДАМИ между тактами (`AddTarget`/`RemoveTarget`) и число
тактов не ограничено заранее -- это другой контракт, под который
`MultiTactSequence` не годится (пересоздавать его на каждый такт с `n_tacts=1`
не работает: он не отдаёт наружу спроецированное следующее `TargetState`, оно
остаётся внутри итератора и терялось бы между тактами). Поэтому `SceneServer`
продвигает каждую цель НАПРЯМУЮ через `Kinematics.project`/`MotionModel.propagate`
-- ТЕ ЖЕ публичные примитивы, что `MultiTactSequence.__next__` вызывает внутри
себя (никакой новой физики, только другой держатель состояния, мутируемый по
командам). `Tact`/`MultiTact` (VO, `tact_sequence.py`) переиспользованы как есть
для формы трек-записи.

M1/M3 (та же логика, что `iter_multi_cubes`, P4/`volume.py`): цели рендерятся
`add_noise=False` и суммируются напрямую, шум добавляется `builder.add_shared_noise`
ОДИН раз поверх суммы -- copy-paste тех 3 строк сюда оправдан тем же аргументом,
что и выше (другой держатель состояния тактов, тот же численный контракт).
"""
from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass, field

import numpy as np

from ..config import JammerFlags, ProjectConfig, SceneConfig
from ..generators import SceneModeler, VolumeBuilder
from ..generators.tact_sequence import MultiTact, Tact
from ..generators.waveforms import AmToCube, LfmToCube, WaveformToCube
from ..graphics import SquareView
from ..motion import Kinematics, MotionModel, TargetState
from .commands import Command, decode_command
from .transport import CMD_TOPIC, Transport

_TO_CUBE: dict[str, WaveformToCube] = {"lfm": LfmToCube(), "am": AmToCube()}


@dataclass
class LiveTarget:
    """Одна ЖИВАЯ цель сервера: изменяемое состояние + модель + свой ГСЧ (M4-подобно).

    `handle_id` -- стабильный идентификатор (НЕ индекс списка -- индексы плавают
    при add/remove), команды `RemoveTarget`/`SetMotion` адресуются по нему.
    """

    handle_id: int
    state: TargetState
    model: MotionModel
    rng: np.random.Generator


@dataclass
class SceneState:
    """Изменяемое состояние сцены сервера -- то, что мутируют `Command.apply` (commands.py)."""

    targets: list[LiveTarget] = field(default_factory=list)
    jammers: JammerFlags = field(default_factory=JammerFlags)
    neighbor_planes: int = 5
    dt: float = 1.0
    _next_id: int = field(default=0, init=False, repr=False)

    def next_id(self) -> int:
        self._next_id += 1
        return self._next_id


class SceneServer:
    """Крутит такты сцены, публикует через `Transport`, принимает команды (Command).

    `transport.subscribe(CMD_TOPIC, ...)` регистрируется в конструкторе -- сервер
    готов принимать команды с первого `step()`. Если `transport` не поддерживает
    `CMD_TOPIC` (например, чистый `WebSocketTransport` без `cmd_bind`), это RuntimeError
    транспорта -- проброшен как есть (см. `ZmqTransport.subscribe`).
    """

    def __init__(self, cfg: ProjectConfig, transport: Transport, state: SceneState,
                 builder: VolumeBuilder | None = None, modeler: SceneModeler | None = None,
                 seed: int | None = None) -> None:
        self._cfg = cfg
        self._kin = Kinematics(cfg)
        self._transport = transport
        self._state = state
        self._builder = builder or VolumeBuilder()
        self._modeler = modeler or SceneModeler()

        seed_seq = np.random.SeedSequence(seed)
        build_seed, jam_seed = seed_seq.spawn(2)
        self._build_rng = np.random.default_rng(build_seed)
        self._jam_rng = np.random.default_rng(jam_seed)

        self._tact = 0
        self._pending: list[Command] = []
        self._pending_lock = threading.Lock()
        self._transport.subscribe(CMD_TOPIC, self._on_command_message)

    @property
    def tact(self) -> int:
        return self._tact

    @property
    def state(self) -> SceneState:
        return self._state

    def _on_command_message(self, _topic: str, _tact: int, payload: object) -> None:
        if not isinstance(payload, dict) or "cmd" not in payload:
            return
        cmd = decode_command(str(payload["cmd"]), dict(payload.get("args") or {}))
        with self._pending_lock:
            self._pending.append(cmd)

    def _apply_pending(self) -> list[Command]:
        with self._pending_lock:
            pending, self._pending = self._pending, []
        for cmd in pending:
            cmd.apply(self._state)
        return pending

    def _cfg_for_tact(self) -> ProjectConfig:
        scene = dataclasses.replace(self._cfg.scene, jammers=self._state.jammers)
        return dataclasses.replace(self._cfg, scene=scene)

    def step(self) -> tuple[MultiTact, np.ndarray] | None:
        """Один такт: применить команды -> продвинуть цели -> собрать объём -> опубликовать.

        `None`, если после применения команд целей не осталось (сервер не падает,
        просто нечего рендерить в этом такте -- `RemoveTarget` увёл сцену в ноль).
        """
        self._apply_pending()
        cfg = self._cfg_for_tact()

        if not self._state.targets:
            self._tact += 1
            return None

        tacts: list[Tact] = []
        vol: np.ndarray | None = None
        for target in self._state.targets:
            sample = self._kin.project(target.state, self._state.dt)
            tacts.append(Tact(state=target.state, sample=sample))
            contrib = self._builder.build_from_sample(sample, cfg, self._build_rng, add_noise=False)
            vol = contrib if vol is None else vol + contrib
            target.state = target.model.propagate(target.state, self._state.dt, target.rng)

        assert vol is not None  # noqa: S101 -- targets непусты, гарантировано циклом выше
        vol = self._builder.add_shared_noise(vol, self._build_rng)
        vol = self._modeler.contribute_to(vol, cfg, self._jam_rng)

        multi_tact = MultiTact(tacts=tuple(tacts))
        to_cube = _TO_CUBE.get(cfg.modulation, _TO_CUBE["lfm"])
        cube = to_cube.fill(vol, cfg)
        square = SquareView(reduce_mode="max", neighbor_planes=self._state.neighbor_planes)
        squares = square.reduce_square(cube)

        self._transport.publish("cube", self._tact, vol)
        self._transport.publish("squares", self._tact, squares)
        self._transport.publish("tracks", self._tact, self._tracks_payload(multi_tact, cfg))

        self._tact += 1
        return multi_tact, vol

    def run(self, n_tacts: int, sleep_s: float = 0.0) -> None:
        """Прогнать `n_tacts` тактов подряд (демо/скрипт-режим). `sleep_s` -- пауза между тактами."""
        import time

        for _ in range(n_tacts):
            self.step()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _tracks_payload(self, multi_tact: MultiTact, cfg: ProjectConfig) -> dict[str, object]:
        """Такт -> примитивы (N2: не numpy/dataclass) -- `{"targets": [...], "jammers": [...]}`."""
        targets = []
        for target, tact in zip(self._state.targets, multi_tact.tacts, strict=True):
            s = tact.sample
            targets.append({
                "id": target.handle_id,
                "kx": s.kx, "ky": s.ky, "r": s.r, "range_bin": s.range_bin,
                "pos": [float(v) for v in tact.state.pos],
                "vel": [float(v) for v in tact.state.vel],
            })
        jammers = self._jammers_payload(cfg.scene)
        return {"targets": targets, "jammers": jammers}

    def _jammers_payload(self, scene: SceneConfig) -> list[dict[str, object]]:
        """Активные помехи такта -> `[{"kind","kx","ky"}, ...]` (пометка углов, SPEC §5)."""
        jammers: list[dict[str, object]] = []
        if scene.jammers.barrage:
            barrage_spec = scene.barrage_spec
            jammers.append({"kind": "barrage", "kx": barrage_spec.kx if barrage_spec else 0.0,
                             "ky": barrage_spec.ky if barrage_spec else 0.0})
        if scene.jammers.comb:
            comb_spec = scene.comb_spec
            jammers.append({"kind": "comb", "kx": comb_spec.kx if comb_spec else 0.0,
                             "ky": comb_spec.ky if comb_spec else 0.0})
        if scene.jammers.ham:
            ham_spec = scene.ham_spec
            jammers.append({"kind": "ham", "kx": ham_spec.kx if ham_spec else 0.0,
                             "ky": ham_spec.ky if ham_spec else 0.0})
        return jammers
