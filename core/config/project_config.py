"""ProjectConfig -- единый агрегат существующих Value Object'ов (P6, F2).

Не дублирует поля `ArrayConfig`/`RangeConfig`/`WaveTimeConfig`/`SceneConfig` --
держит на них ссылки. Новые поля -- только то, чего в существующем стеке нет
(ветка модуляции, окно/шаг АМ, число импульсов, транспорт, визуал-закладка).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .array_config import ArrayConfig, RangeConfig
from .scene_config import SceneConfig
from .waveform_config import WaveTimeConfig

_VALID_MODULATIONS = ("lfm", "am")
_VALID_AM_STEPS = (8, 16, 32, 64)


@dataclass(frozen=True)
class ProjectConfig:
    """Единая точка правды прогона P6: геометрия + окно/шаг + транспорт + визуал.

    Агрегирует существующие `ArrayConfig`/`RangeConfig`/`WaveTimeConfig`/`SceneConfig`
    (F2) -- не заменяет их. Загрузка/сохранение -- через `data_context.run_workspace`
    (`config_to_dict`/`to_yaml`/`from_yaml`), не `YamlConfigSource` (A5).
    """

    array: ArrayConfig = field(default_factory=ArrayConfig)
    range_: RangeConfig = field(default_factory=RangeConfig)
    wave: WaveTimeConfig = field(default_factory=WaveTimeConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)

    # --- P6: ветвление ЛЧМ/АМ + рантайм-параметры ---------------------------
    modulation: str = "lfm"          # "lfm" | "am" (выбор фронтенда, SPEC §1/§2)
    am_window_depth: int = 16        # D (16..256), только АМ (локальный 3D-FFT окно nx×ny×D)
    am_step: int = 8                 # 8/16/32/64, только АМ (дефолт D/2 -> overlap 50%)
    n_pulses: int = 64               # slow-time (Доплер, заглушка P6)
    transport_endpoint: str = "tcp://127.0.0.1:5556"
    viz_neighbor_planes: int = 5     # закладка +-N плоскостей вокруг объекта (SPEC §5)

    def __post_init__(self) -> None:
        if self.modulation not in _VALID_MODULATIONS:
            raise ValueError(
                f"modulation должна быть одной из {_VALID_MODULATIONS}, получено {self.modulation!r}"
            )
        if not (16 <= self.am_window_depth <= 256):
            raise ValueError(
                f"am_window_depth должен быть в [16, 256], получено {self.am_window_depth}"
            )
        if self.am_step not in _VALID_AM_STEPS:
            raise ValueError(f"am_step должен быть одним из {_VALID_AM_STEPS}, получено {self.am_step}")
        if self.n_pulses < 1:
            raise ValueError("n_pulses должен быть положительным")
        if self.viz_neighbor_planes < 0:
            raise ValueError("viz_neighbor_planes не может быть отрицательным")
