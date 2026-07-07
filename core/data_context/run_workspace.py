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
from typing import Any

from ..config import SimulationConfig

__all__ = ["RunWorkspace", "config_to_dict", "to_yaml", "from_yaml"]


# --- сериализация конфигурации сцены -> простой dict (воспроизводимость) -------
def config_to_dict(cfg: SimulationConfig) -> dict[str, Any]:
    """SimulationConfig -> вложенный dict со всеми параметрами источников."""
    return {
        "seed": cfg.seed,
        "array": {"nx": cfg.array.nx, "ny": cfg.array.ny},
        "range": {"n_real": cfg.range.n_real, "n_fft": cfg.range.n_fft},
        "scene": {
            "emitters": [_emitter_to_dict(e) for e in cfg.scene.emitters],
            "thermal": {"power": cfg.scene.thermal.power},
        },
    }


def _emitter_to_dict(emitter: Any) -> dict[str, Any]:
    """Спецификация излучателя -> dict с сохранением её типа (для восстановления)."""
    data: dict[str, Any] = {"type": type(emitter).__name__}
    data.update(asdict(emitter))
    return data


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
