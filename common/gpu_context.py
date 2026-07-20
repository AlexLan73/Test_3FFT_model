# vendored+адаптировано from DSP-GPU/DSP/Python/common/gpu_context.py (2026-07-14).
#
# Отличие от оригинала: DSP-GPU использует свой `GPULoader` (перебор build/-каталогов
# чужого репо) — здесь вместо него `core.gpu_libs.loader` (копии .so синкаются в
# core/gpu_libs/, R1 спеки §2.2.1). Оставлен только ROCm-путь (`get_rocm`) — OpenCL
# (`GPUContext`/`get`) в radar3d не используется (HipBackend — только ROCm, §4.3).
"""
gpu_context.py — GPUContextManager Singleton
=============================================

Singleton (GoF):
  Создаёт GPU-контекст один раз для всей сессии.
  Переиспользование контекста критично — создание занимает ~1-2 сек.

  GPU-индекс берётся из `core/gpu_libs/configGPU.json`, секция "gpus" →
  первый элемент с "is_active": true.

Usage:
    ctx = GPUContextManager.get_rocm()   # ROCmGPUContext или None, если недоступен
"""
from __future__ import annotations

import warnings

from core.gpu_libs import loader as gpu_libs

from .gpu_configs import first_active_gpu_id


def _active_device() -> int:
    """Вернуть id первого активного GPU из `core/gpu_libs/configGPU.json`."""
    cfg = gpu_libs.config_path()
    if not cfg.exists():
        return 0
    return first_active_gpu_id(cfg, default=0)


class GPUContextManager:
    """Singleton — хранит ROCmGPUContext для всей сессии.

    Attributes:
        _rocm_context:  единственный ROCmGPUContext (или None, если недоступен)
        _device_index:  индекс GPU на котором создан контекст
    """

    _rocm_context = None
    _device_index: int = 0
    _create_attempted: bool = False

    @classmethod
    def get_rocm(cls, device: int | None = None):
        """Получить или создать ROCmGPUContext.

        Args:
            device: индекс GPU. None = из configGPU.json.

        Returns:
            ROCmGPUContext или None если ROCm/`.so` недоступны.
        """
        if not cls._create_attempted:
            cls._create_attempted = True
            cls._device_index = device if device is not None else _active_device()
            cls._try_create_rocm(cls._device_index)
        return cls._rocm_context

    @classmethod
    def _try_create_rocm(cls, device: int) -> None:
        """Создать ROCmGPUContext через dsp_core (core.gpu_libs.loader)."""
        if not gpu_libs.available():
            cls._rocm_context = None
            return
        try:
            dsp_core = gpu_libs.load("dsp_core")
            if not hasattr(dsp_core, "ROCmGPUContext"):
                cls._rocm_context = None
                return
            cls._rocm_context = dsp_core.ROCmGPUContext(device)
        except Exception as exc:  # GPU-init может падать по многим причинам (ROCm/HIP ошибки)
            warnings.warn(
                f"[GPUContextManager] dsp_core.ROCmGPUContext(device={device}) failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            cls._rocm_context = None

    @classmethod
    def is_rocm_available(cls) -> bool:
        """True если ROCm-контекст создан."""
        return cls.get_rocm() is not None

    @classmethod
    def device_index(cls) -> int:
        """Индекс GPU на котором создан контекст."""
        return cls._device_index

    @classmethod
    def reset(cls) -> None:
        """Сбросить контекст (для тестирования GPUContextManager)."""
        cls._rocm_context = None
        cls._create_attempted = False
