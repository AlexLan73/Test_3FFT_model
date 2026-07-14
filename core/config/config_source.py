"""ConfigSource — абстрактный ввод конфигов генерации (Strategy/Facade, P0).

Грузит `configs/*.yaml` в существующие/новые VO (`WaveTimeConfig`). НЕ второй
конфиг-слой (R10 спеки) — просто источник данных для уже существующих Value Object.

`iter_configs()` — sweep-хук (ответ Alex Q2): по умолчанию отдаёт один конфиг,
но интерфейс уже готов к перебору датасета (несколько YAML/вариаций).
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from .array_config import ArrayConfig
from .waveform_config import WaveTimeConfig


class ConfigSource(Protocol):
    """Strategy: источник `WaveTimeConfig`."""

    def load(self) -> WaveTimeConfig:
        """Единственный конфиг (реализуют конкретные источники)."""
        ...

    def iter_configs(self) -> Iterator[WaveTimeConfig]:
        """Перебор конфигов для датасета. По умолчанию — один (`load()`)."""
        yield self.load()


class DefaultConfigSource(ConfigSource):
    """Baseline-дефолты §5.1 без внешних зависимостей (R10: работает без pyyaml, Windows-путь)."""

    def load(self) -> WaveTimeConfig:
        return WaveTimeConfig()


class YamlConfigSource(ConfigSource):
    """Грузит `WaveTimeConfig` из YAML-файла (pyyaml)."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> WaveTimeConfig:
        try:
            import yaml
        except ImportError as exc:
            # R10: pyyaml может отсутствовать (Windows/минимальный venv) — понятная ошибка,
            # а не голый ModuleNotFoundError. Альтернатива — DefaultConfigSource.
            raise ImportError(
                "pyyaml не установлен — YamlConfigSource недоступен. "
                "Используйте DefaultConfigSource (baseline §5.1 без YAML, Windows-путь)."
            ) from exc

        with self._path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        array_raw = raw.get("array", {})
        array = ArrayConfig(
            nx=int(array_raw.get("nx", 16)),
            ny=int(array_raw.get("ny", 16)),
        )
        return WaveTimeConfig(
            fs=float(raw.get("fs", 12e6)),
            carrier_hz=float(raw.get("carrier_hz", 2e6)),
            fdev_hz=float(raw.get("fdev_hz", 6e6)),
            n_samples=int(raw.get("n_samples", 8192)),
            array=array,
            seed=int(raw.get("seed", 7)),
        )
