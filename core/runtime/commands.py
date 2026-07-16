"""Command -- команды панель->сервер (Command pattern, P6), сериализуемые msgpack.

Каждая команда: (1) чистые примитивные поля (msgpack-совместимые, N2), (2)
`to_message()` -> `(cmd_name, args)` для `codec.encode_command`, (3) `apply(state)`
-- детерминированно меняет `SceneState` (см. `scene_server.py`). Применяются
СЕРВЕРОМ на следующем такте (`SceneServer._apply_pending`, не мгновенно -- команды
приходят асинхронно из фонового треда транспорта).

Реестр `COMMAND_REGISTRY` + `decode_command(cmd_name, args)` -- десериализация
входящего `{"cmd": ..., "args": ...}` (то, что кладёт `codec.decode` для топика
`CMD_TOPIC`) обратно в объект команды, без `pickle`/`eval` (N2: только явный
маппинг имя -> класс).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .scene_server import SceneState

_MOTION_NAMES = ("cv", "markov", "turn", "accel", "weave")


class Command(Protocol):
    """Command pattern: `apply` мутирует состояние сцены, `to_message` -- для отправки."""

    def apply(self, state: SceneState) -> None: ...

    def to_message(self) -> tuple[str, dict[str, Any]]: ...


def _build_motion(name: str, **kwargs: float):
    """Строит `MotionModel` по имени (N2: команда несёт ИМЯ, не python-класс)."""
    from core.motion import ConstantAccel, ConstantVelocity, CoordinatedTurn, MarkovDrift, WeavingManeuver

    factory = {
        "cv": ConstantVelocity,
        "markov": MarkovDrift,
        "turn": CoordinatedTurn,
        "accel": ConstantAccel,
        "weave": WeavingManeuver,
    }
    cls = factory.get(name)
    if cls is None:
        raise ValueError(f"неизвестная модель движения {name!r}, ожидалось одно из {_MOTION_NAMES}")
    return cls(**kwargs)


@dataclass(frozen=True)
class AddTarget:
    """Добавить новую цель: старт `pos`/`vel` (м, м/с) + имя закона движения."""

    pos: tuple[float, float, float]
    vel: tuple[float, float, float]
    motion: str = "cv"
    seed: int | None = None

    def apply(self, state: SceneState) -> None:
        import numpy as np

        from core.motion import TargetState

        from .scene_server import LiveTarget

        target_state = TargetState(pos=np.asarray(self.pos, dtype=np.float64),
                                   vel=np.asarray(self.vel, dtype=np.float64))
        model = _build_motion(self.motion)
        rng = np.random.default_rng(self.seed)
        state.targets.append(
            LiveTarget(handle_id=state.next_id(), state=target_state, model=model, rng=rng)
        )

    def to_message(self) -> tuple[str, dict[str, Any]]:
        args: dict[str, Any] = {
            "pos": list(self.pos), "vel": list(self.vel), "motion": self.motion,
        }
        if self.seed is not None:
            args["seed"] = self.seed
        return "add_target", args


@dataclass(frozen=True)
class RemoveTarget:
    """Убрать цель по стабильному `handle_id` (НЕ по индексу -- индексы плавают при add/remove)."""

    handle_id: int

    def apply(self, state: SceneState) -> None:
        state.targets = [t for t in state.targets if t.handle_id != self.handle_id]

    def to_message(self) -> tuple[str, dict[str, Any]]:
        return "remove_target", {"handle_id": self.handle_id}


@dataclass(frozen=True)
class SetMotion:
    """Сменить закон движения уже существующей цели (`handle_id`) на лету."""

    handle_id: int
    motion: str

    def apply(self, state: SceneState) -> None:
        model = _build_motion(self.motion)
        for target in state.targets:
            if target.handle_id == self.handle_id:
                target.model = model
                return

    def to_message(self) -> tuple[str, dict[str, Any]]:
        return "set_motion", {"handle_id": self.handle_id, "motion": self.motion}


@dataclass(frozen=True)
class EnableJammer:
    """Частичное включение/выключение помех такта (`None` поле -- не трогать)."""

    barrage: bool | None = None
    comb: bool | None = None
    ham: bool | None = None

    def apply(self, state: SceneState) -> None:
        import dataclasses

        updates = {k: v for k, v in
                   (("barrage", self.barrage), ("comb", self.comb), ("ham", self.ham))
                   if v is not None}
        if updates:
            state.jammers = dataclasses.replace(state.jammers, **updates)

    def to_message(self) -> tuple[str, dict[str, Any]]:
        args = {k: v for k, v in
                (("barrage", self.barrage), ("comb", self.comb), ("ham", self.ham))
                if v is not None}
        return "enable_jammer", args


@dataclass(frozen=True)
class Step:
    """Задать длительность такта `dt` (с) для последующих шагов сервера."""

    dt: float = 1.0

    def apply(self, state: SceneState) -> None:
        if self.dt <= 0:
            raise ValueError(f"Step.dt должен быть положительным, получено {self.dt}")
        state.dt = self.dt

    def to_message(self) -> tuple[str, dict[str, Any]]:
        return "step", {"dt": self.dt}


@dataclass(frozen=True)
class SetNeighborPlanes:
    """Закладка +-N плоскостей (SPEC §5): меняет `neighbor_planes` сцены/панели."""

    n: int = 5

    def apply(self, state: SceneState) -> None:
        if self.n < 0:
            raise ValueError(f"SetNeighborPlanes.n не может быть отрицательным, получено {self.n}")
        state.neighbor_planes = self.n

    def to_message(self) -> tuple[str, dict[str, Any]]:
        return "set_neighbor_planes", {"n": self.n}


COMMAND_REGISTRY: dict[str, type] = {
    "add_target": AddTarget,
    "remove_target": RemoveTarget,
    "set_motion": SetMotion,
    "enable_jammer": EnableJammer,
    "step": Step,
    "set_neighbor_planes": SetNeighborPlanes,
}


def decode_command(cmd_name: str, args: dict[str, Any]) -> Command:
    """`(cmd_name, args)` (из `codec.decode`) -> объект команды (реестр, без eval/pickle)."""
    cls = COMMAND_REGISTRY.get(cmd_name)
    if cls is None:
        raise ValueError(f"неизвестная команда {cmd_name!r}, ожидалось одно из {sorted(COMMAND_REGISTRY)}")
    kwargs = dict(args)
    if cls is AddTarget:
        kwargs["pos"] = tuple(kwargs["pos"])
        kwargs["vel"] = tuple(kwargs["vel"])
    return cls(**kwargs)  # type: ignore[no-any-return]
