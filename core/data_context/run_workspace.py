"""Каталог одного прогона с градацией по дате/времени + manifest сцены.

out/runs/<YYYY-MM-DD>/<HHMMSS>/<сцена>/{manifest.yaml, figures/, data/}

RunWorkspace -- Pure Fabrication (GRASP): отвечает только за раскладку файлов
прогона. Сериализация сцены -> dict + минимальный YAML без внешних зависимостей
(PyYAML в офлайн-среде нет; данные простые -> свой дампер, контролируем сами).
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from ..config import (
    ArrayConfig,
    BarrageSpec,
    DrfmCombSpec,
    EmitterSpec,
    HamEmitterSpec,
    JammerFlags,
    ProjectConfig,
    RangeConfig,
    SceneConfig,
    SimulationConfig,
    TargetSpec,
    ThermalNoiseSpec,
    WaveTimeConfig,
)

_EMITTER_TYPES: dict[str, type[EmitterSpec]] = {
    "TargetSpec": TargetSpec,
    "DrfmCombSpec": DrfmCombSpec,
    "BarrageSpec": BarrageSpec,
    "HamEmitterSpec": HamEmitterSpec,
    "EmitterSpec": EmitterSpec,
}

__all__ = ["RunWorkspace", "config_to_dict", "to_yaml", "from_yaml",
           "project_config_to_dict", "project_config_from_dict"]


# --- сериализация конфигурации сцены -> простой dict (воспроизводимость) -------
def config_to_dict(cfg: SimulationConfig | ProjectConfig) -> dict[str, Any]:
    """SimulationConfig/ProjectConfig -> вложенный dict со всеми параметрами.

    A5: `ProjectConfig` (P6, агрегат) грузится/сохраняется здесь же -- НЕ через
    `YamlConfigSource` (требует PyYAML, которого в офлайн-среде нет, и знает
    только про `WaveTimeConfig`).
    """
    if isinstance(cfg, ProjectConfig):
        return project_config_to_dict(cfg)
    return {
        "seed": cfg.seed,
        "array": {"nx": cfg.array.nx, "ny": cfg.array.ny},
        "range": {"n_real": cfg.range.n_real, "n_fft": cfg.range.n_fft},
        "scene": _scene_to_dict(cfg.scene),
    }


def _scene_to_dict(scene: SceneConfig) -> dict[str, Any]:
    """SceneConfig -> dict. H2: сериализует ВСЕ поля (не только emitters+thermal) --
    иначе `jammers`/`barrage_spec`/`comb_spec`/`ham_spec` теряются на роундтрипе
    (см. `project_config_from_dict`, парная сборка ниже)."""
    data: dict[str, Any] = {
        "emitters": [_emitter_to_dict(e) for e in scene.emitters],
        "thermal": {"power": scene.thermal.power},
        "jammers": asdict(scene.jammers),
    }
    if scene.barrage_spec is not None:
        data["barrage_spec"] = _emitter_to_dict(scene.barrage_spec)
    if scene.comb_spec is not None:
        data["comb_spec"] = _emitter_to_dict(scene.comb_spec)
    if scene.ham_spec is not None:
        data["ham_spec"] = _emitter_to_dict(scene.ham_spec)
    return data


def project_config_to_dict(cfg: ProjectConfig) -> dict[str, Any]:
    """`ProjectConfig` -> dict (свой YAML-дампер, без PyYAML, A5)."""
    return {
        "array": {"nx": cfg.array.nx, "ny": cfg.array.ny},
        "range": {"n_real": cfg.range_.n_real, "n_fft": cfg.range_.n_fft},
        "wave": {
            "fs": cfg.wave.fs,
            "carrier_hz": cfg.wave.carrier_hz,
            "fdev_hz": cfg.wave.fdev_hz,
            "n_samples": cfg.wave.n_samples,
            "array": {"nx": cfg.wave.array.nx, "ny": cfg.wave.array.ny},
            "seed": cfg.wave.seed,
        },
        "scene": _scene_to_dict(cfg.scene),
        "modulation": cfg.modulation,
        "am_window_depth": cfg.am_window_depth,
        "am_step": cfg.am_step,
        "n_pulses": cfg.n_pulses,
        "transport_endpoint": cfg.transport_endpoint,
        "viz_neighbor_planes": cfg.viz_neighbor_planes,
    }


def project_config_from_dict(data: dict[str, Any]) -> ProjectConfig:
    """Парная сборка `ProjectConfig` из dict, произведённого `project_config_to_dict`."""
    array_raw = data.get("array", {})
    range_raw = data.get("range", {})
    wave_raw = data.get("wave", {})
    wave_array_raw = wave_raw.get("array", {})
    scene_raw = data.get("scene", {})

    jammers_raw = scene_raw.get("jammers", {})
    scene = SceneConfig(
        emitters=tuple(_emitter_from_dict(e) for e in scene_raw.get("emitters", [])),
        thermal=ThermalNoiseSpec(power=float(scene_raw.get("thermal", {}).get("power", 0.02))),
        jammers=JammerFlags(
            barrage=bool(jammers_raw.get("barrage", False)),
            comb=bool(jammers_raw.get("comb", False)),
            ham=bool(jammers_raw.get("ham", False)),
            cw=bool(jammers_raw.get("cw", False)),
            vfd=bool(jammers_raw.get("vfd", False)),
            arc=bool(jammers_raw.get("arc", False)),
            clutter=bool(jammers_raw.get("clutter", False)),
        ),
        barrage_spec=cast(BarrageSpec, _emitter_from_dict(scene_raw["barrage_spec"]))
        if "barrage_spec" in scene_raw else None,
        comb_spec=cast(DrfmCombSpec, _emitter_from_dict(scene_raw["comb_spec"]))
        if "comb_spec" in scene_raw else None,
        ham_spec=cast(HamEmitterSpec, _emitter_from_dict(scene_raw["ham_spec"]))
        if "ham_spec" in scene_raw else None,
    )
    wave = WaveTimeConfig(
        fs=float(wave_raw.get("fs", 12e6)),
        carrier_hz=float(wave_raw.get("carrier_hz", 2e6)),
        fdev_hz=float(wave_raw.get("fdev_hz", 6e6)),
        n_samples=int(wave_raw.get("n_samples", 8192)),
        array=ArrayConfig(nx=int(wave_array_raw.get("nx", 16)), ny=int(wave_array_raw.get("ny", 16))),
        seed=int(wave_raw.get("seed", 7)),
    )
    return ProjectConfig(
        array=ArrayConfig(nx=int(array_raw.get("nx", 16)), ny=int(array_raw.get("ny", 16))),
        range_=RangeConfig(n_real=int(range_raw.get("n_real", 16)), n_fft=int(range_raw.get("n_fft", 16))),
        wave=wave,
        scene=scene,
        modulation=str(data.get("modulation", "lfm")),
        am_window_depth=int(data.get("am_window_depth", 16)),
        am_step=int(data.get("am_step", 8)),
        n_pulses=int(data.get("n_pulses", 64)),
        transport_endpoint=str(data.get("transport_endpoint", "tcp://127.0.0.1:5556")),
        viz_neighbor_planes=int(data.get("viz_neighbor_planes", 5)),
    )


def _emitter_to_dict(emitter: Any) -> dict[str, Any]:
    """Спецификация излучателя -> dict с сохранением её типа (для восстановления)."""
    data: dict[str, Any] = {"type": type(emitter).__name__}
    data.update(asdict(emitter))
    return data


def _emitter_from_dict(data: dict[str, Any]) -> EmitterSpec:
    """Парная сборка спецификации излучателя (по метке `type`, см. `_emitter_to_dict`)."""
    payload = dict(data)
    type_name = payload.pop("type", "EmitterSpec")
    cls = _EMITTER_TYPES.get(type_name, EmitterSpec)
    return cls(**payload)


# --- минимальный YAML-дампер (подмножество: dict / list / скаляры) ------------
def to_yaml(obj: Any, indent: int = 0) -> str:
    """Сериализует dict/list/скаляр в валидный YAML. Без внешних зависимостей."""
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return pad + "{}\n"
        out = ""
        for key, val in obj.items():
            if isinstance(val, (dict, list)) and val:
                out += f"{pad}{key}:\n{to_yaml(val, indent + 1)}"
            else:
                out += f"{pad}{key}: {_scalar(val)}\n"
        return out
    if isinstance(obj, list):
        if not obj:
            return pad + "[]\n"
        out = ""
        for item in obj:
            if isinstance(item, dict) and item:
                # ключи элемента печатаем с отступом indent+1, первый -- на "- "
                lines = to_yaml(item, indent + 1).rstrip("\n").split("\n")
                prefix = "  " * (indent + 1)
                out += f"{pad}- {lines[0][len(prefix):]}\n"
                for line in lines[1:]:
                    out += line + "\n"
            else:
                out += f"{pad}- {_scalar(item)}\n"
        return out
    return pad + _scalar(obj) + "\n"


def _scalar(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if val is None:
        return "null"
    if isinstance(val, float):
        return repr(round(val, 6))
    return str(val)


# --- минимальный YAML-загрузчик (парный к to_yaml, то же подмножество) ---------
def from_yaml(text: str) -> Any:
    """Разбирает YAML, выданный to_yaml (dict/list/скаляры). Без зависимостей."""
    rows: list[list[Any]] = []
    for line in text.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append([len(line) - len(line.lstrip(" ")), line.strip()])
    if not rows:
        return {}
    value, _ = _parse_node(rows, 0)
    return value


def _parse_node(rows: list[list[Any]], i: int) -> tuple[Any, int]:
    indent = rows[i][0]
    if rows[i][1].startswith("- "):                         # список
        items: list[Any] = []
        while i < len(rows) and rows[i][0] == indent and rows[i][1].startswith("- "):
            content = rows[i][1][2:]
            if ": " in content or content.endswith(":"):    # элемент-словарь
                sub = [[indent + 2, content]]
                i += 1
                while i < len(rows) and rows[i][0] > indent:
                    sub.append(rows[i])
                    i += 1
                node, _ = _parse_node(sub, 0)
                items.append(node)
            else:
                items.append(_load_scalar(content))
                i += 1
        return items, i
    node_d: dict[str, Any] = {}                             # словарь
    while i < len(rows) and rows[i][0] == indent and not rows[i][1].startswith("- "):
        key, _, rest = rows[i][1].partition(":")
        key, rest = key.strip(), rest.strip()
        if rest:
            node_d[key] = _load_scalar(rest)
            i += 1
        else:
            child, i = _parse_node(rows, i + 1)
            node_d[key] = child
    return node_d, i


def _load_scalar(s: str) -> Any:
    if s in ("true", "false"):
        return s == "true"
    if s in ("null", "{}", "[]"):
        return None if s == "null" else ({} if s == "{}" else [])
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


# --- workspace прогона ---------------------------------------------------------
class RunWorkspace:
    """Каталог одного прогона: out/runs/<дата>/<время>/. Один запуск = один run."""

    def __init__(self, out_root: str = "./out", stamp: datetime | None = None):
        now = stamp or datetime.now()
        self.date = now.strftime("%Y-%m-%d")
        self.time = now.strftime("%H%M%S")
        self.run_id = f"{self.date}/{self.time}"
        self.base = Path(out_root) / "runs" / self.date / self.time
        self.base.mkdir(parents=True, exist_ok=True)
        self._update_latest()

    def _update_latest(self) -> None:
        """out/runs/latest -> текущий прогон (стабильный путь для документации)."""
        link = self.base.parent.parent / "latest"          # out/runs/latest
        target = Path(self.date) / self.time                # относительная цель
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        except OSError:
            pass                                            # ФС без симлинков (напр. Windows без прав)

    def figures_dir(self, scene: str) -> str:
        path = self.base / scene / "figures"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def data_dir(self, scene: str) -> str:
        path = self.base / scene / "data"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def write_manifest(self, scene: str, payload: dict[str, Any]) -> str:
        path = self.base / scene / "manifest.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(to_yaml(payload), encoding="utf-8")
        return str(path)

    def write_summary(self, text: str, name: str = "summary.md") -> str:
        path = self.base / name
        path.write_text(text, encoding="utf-8")
        return str(path)
