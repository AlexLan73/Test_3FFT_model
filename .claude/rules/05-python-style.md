# 05 — Python Style (radar3d)

> **paths:** `core/**`, `common/**`, `tests/**`, `main.py`, `*.py`

## Базовое

- Python ≥ 3.11.
- **Pathlib** для путей (`from pathlib import Path`). Минимум `os.path.join`.
- **Type hints** везде (`def f(x: int) -> str:`). `from __future__ import annotations` для forward-refs.
- Имена пакетов/каталогов — **строчные ASCII** (`core`, не `Core`): Linux ФС регистрозависима.
- Теней переменных избегать.

## SOLID + GoF (как в проекте)

| Принцип | Применение в radar3d |
|---------|----------------------|
| **S**ingle responsibility | классификация ≠ спектральное преобразование (`models/classification/`) |
| **O**pen-closed | новая помеха = подкласс `SignalSource` + `factory.register(...)`, тракт не трогаем |
| **L**iskov | `Cnn3DClassifier` заменяет `RuleBasedClassifier` — тот же интерфейс |
| **I**nterface segregation | маленькие протоколы (`Protocol`) для роли |
| **D**ependency injection | связывание в `main.py` (Composition Root), без global'ов |

Паттерны: Strategy (`WindowFunction`/`RadarModel`/`Visualizer`/`CubeClassifier`),
Composite (`Scene`), Abstract Factory/Registry (`EmitterFactory`), Builder
(`SceneBuilder`), Template Method (`RadarModel.process`), Facade (`DataContext`),
Value Object (конфиги, `SpectralCube`, `Axis`).

## Naming

- `snake_case` — функции/переменные.
- `PascalCase` — классы.
- `UPPER_CASE` — константы (`MAX_RANGE = 40`).
- Префикс `_` — приватное.
- Префикс `I` НЕ используем (не Java) — пишем `CubeClassifier(Protocol)`.

## Структура пакета

```python
# core/models/__init__.py — реэкспорт публичного API подпакета
from .fft3d import Fft3DModel
from .windows import AxisWindows, HannWindow
from .classification import RuleBasedClassifier, Cnn3DClassifier

__all__ = ["Fft3DModel", "AxisWindows", "HannWindow",
           "RuleBasedClassifier", "Cnn3DClassifier"]
```

## NumPy

- Векторизация вместо python-циклов где возможно.
- Явный `dtype` (`np.float32` / `np.complex64`) — экономия памяти под GPU-перенос.
- Не мутировать входные массивы — возвращать новые (чистота).

## GPU / torch-ROCm

- **venv = Python 3.12** (не 3.13!): офлайн-колёса torch собраны под `cp312`.
- torch **2.11.0+rocm7.2** ставится из офлайн-пака `--no-deps`:
  `/mnt/data/offline-debian-pack/3_python_wheels/torch-rocm/torch-2.11.0+rocm7.2-cp312-cp312-*.whl`
- `triton-rocm` **не нужен** — он только для `torch.compile`/inductor. Conv3d/CNN
  бегут в eager-режиме на ROCm. (Колесо требует ровно `triton-rocm==3.6.0`, которого
  нет → ставим torch через `--no-deps`.)
- GPU: AMD Radeon RX 9070 (gfx1201), ROCm 7.2. Проверка: `torch.cuda.is_available()`.
- Эталон рабочего torch-стека — `/home/alex/finetune-env/.venv` (Python 3.12).
- Тяжёлые скачивания (>600 МБ) — НЕ качать без спроса; torch уже есть локально.

## Запреты

- `from xxx import *`
- bare `except:` (нужен `except Exception as e`)
- `print()` в библиотечном коде (в `main.py`/демо — допустимо)
- mutable default args (`def f(x=[])`)
- `lambda` для нетривиальной логики (>30 символов)

## Линт

```bash
ruff check core/ common/ tests/
mypy core/
```

`pyproject.toml` настроен: line-length=110, ruff rules E/F/W/I/N/UP/B/SIM/RET.
