# vendored (частично) from DSP-GPU/DSP/Python/common/configs.py (2026-07-14).
"""Подмножество `configs.py` DSP-GPU — только парсинг `configGPU.json` (§8 спеки).

Оригинал содержит ещё `SignalConfig`/`HeterodyneConfig`/`FilterConfig`/`ProcessorConfig` —
они не нужны `GPUContextManager` (не плодим лишнее, правило 01 «не плодить сущности»).
Взято без изменений: `GpuEntry`/`load_gpu_config`/`active_gpu_ids`/`first_active_gpu_id`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GpuEntry:
    """Одна запись GPU из configGPU.json."""
    id: int = 0
    is_active: bool = False
    name: str = ""


def load_gpu_config(config_path: str | Path) -> list[GpuEntry]:
    """Прочитать configGPU.json и вернуть список GpuEntry.

    Args:
        config_path: путь к configGPU.json (рядом с бинарником/либой).

    Returns:
        Список GpuEntry из секции "gpus". Пустой список если файл не найден.
    """
    path = Path(config_path)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        entries = []
        for gpu in data.get("gpus", []):
            entries.append(GpuEntry(
                id=int(gpu.get("id", 0)),
                is_active=bool(gpu.get("is_active", False)),
                name=str(gpu.get("name", "")),
            ))
        return entries
    except Exception:
        return []


def active_gpu_ids(config_path: str | Path) -> list[int]:
    """Вернуть список id активных GPU из configGPU.json."""
    return [e.id for e in load_gpu_config(config_path) if e.is_active]


def first_active_gpu_id(config_path: str | Path, default: int = 0) -> int:
    """Вернуть id первого активного GPU (default если конфиг не найден/пустой)."""
    ids = active_gpu_ids(config_path)
    return ids[0] if ids else default
