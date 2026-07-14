"""Тонкий loader GPU `.so` (R1, спека §2.2.1 signal_generators_2026-07-13.md).

Копии `dsp_*.so` (cp313) DSP-GPU лежат прямо в этом каталоге (`core/gpu_libs/`),
рядом с `configGPU.json` (его ищет `common.gpu_context.GPUContextManager`).
Не трогаем `sys.path` чужого репо (DSP-GPU) — хрупко; синкаем копии сюда
(`sync_gpu_libs.sh`).

На Windows / без собранных `.so` (или под Python != 3.13, ABI не совпадёт) —
`available()` возвращает False, `require()`/`load()` бросают `GpuLibsUnavailableError` —
понятный сигнал для `SkipTest` в тестах (правило 04).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

_LIBS_DIR = Path(__file__).resolve().parent

# Порядок — как в спеке §2.2.1 (CW/ЛЧМ-генератор первым; остальные — задел P4+/§6).
MODULE_NAMES: tuple[str, ...] = (
    "dsp_core",
    "dsp_signal_generators",
    "dsp_heterodyne",
    "dsp_radar",
    "dsp_spectrum",
)


class GpuLibsUnavailableError(RuntimeError):
    """GPU `.so` недоступны: нет файлов, не Linux, не cp313, либо ROCm не поднят."""


def _ensure_path() -> None:
    path_str = str(_LIBS_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def available() -> bool:
    """True, если `dsp_core` реально импортируется (не проверяет наличие ROCm-девайса)."""
    _ensure_path()
    try:
        importlib.import_module("dsp_core")
    except ImportError:
        return False
    return True


def load(name: str) -> ModuleType:
    """Импортировать один из `dsp_*` модулей (см. `MODULE_NAMES`).

    Raises:
        ValueError: неизвестное имя модуля.
        GpuLibsUnavailableError: модуль не импортируется (нет `.so`/несовпадение ABI).
    """
    if name not in MODULE_NAMES:
        raise ValueError(f"Неизвестный GPU-модуль {name!r}, доступны: {MODULE_NAMES}")
    _ensure_path()
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise GpuLibsUnavailableError(f"{name} недоступен: {exc}") from exc


def config_path() -> Path:
    """Путь к `configGPU.json` рядом с `.so` (для `GPUContextManager`)."""
    return _LIBS_DIR / "configGPU.json"


def require() -> None:
    """Бросить `GpuLibsUnavailableError` с понятным сообщением, если GPU-стек недоступен."""
    if not available():
        raise GpuLibsUnavailableError(
            "core/gpu_libs/*.so не найдены или не импортируются — нужен Linux + "
            "Python 3.13 (cp313 ABI) + собранные dsp_*.so (см. core/gpu_libs/sync_gpu_libs.sh)."
        )
