# 📁 Описание папок проекta radar3d

Карта каталогов репозитория с назначением каждого. Имена пакетов — строчные
ASCII (PEP 8): на Linux ФС регистрозависима, заглавная папка ломает импорты.

## Корень

| Путь | Назначение |
|------|------------|
| `main.py` | точка входа: эталонный прогон (Composition Root) → `out/` |
| `classify_demo.py` | демо детерминированной классификации (без torch) |
| `train_cnn.py` | обучение 3D-CNN → `cnn3d.pt` (нужен torch+ROCm) |
| `pyproject.toml` | метаданные пакета, deps, конфиг ruff/mypy |
| `README.md` | обзор проекта, запуск, паттерны |
| `CLAUDE.md` | инструкции ассистента (Кодо), архитектура, команды |
| `.gitignore` | игнор артефактов (`out/`, веса, MCP-память, worktrees) |

## `core/` — ядро (движок обработки)

| Подпакет | Назначение | Ключевые файлы |
|----------|------------|----------------|
| `core/config/` | неизменяемые параметры прогона (Value Objects) | `array_config.py`, `scene_config.py`, `simulation_config.py` |
| `core/generators/` | синтез сырого куба из сцены | `sources.py`, `jammers.py`, `scene.py`, `factory.py`, `grid.py` |
| `core/models/` | спектральное преобразование (3D-БПФ) | `base.py`, `fft3d.py`, `windows.py`, `result.py` |
| `core/models/classification/` | классификация куба | `classifier.py`, `rule_based.py`, `cnn3d.py`, `dataset.py`, `labels.py` |
| `core/graphics/` | визуализаторы + запись фигур | `visualizer.py`, `cube_scatter.py`, `angular_map.py`, `range_profile.py`, `writer.py` |
| `core/data_context/` | хранение кубов (Facade + Repository) | `data_context.py`, `repository.py` |
| `core/controller.py` | координатор прогона (GRASP Controller) | — |

## `common/` — общая инфраструктура

| Файл | Назначение |
|------|------------|
| `common/runner.py` | `TestRunner`/`AssertionGroup`/`SkipTest` — **замена pytest** (правило 04) |

## `tests/` — тесты (НЕ pytest)

| Файл | Назначение |
|------|------------|
| `tests/all_test.py` | агрегатор всех наборов: `python tests/all_test.py` |
| `tests/test_smoke.py` | сквозной smoke: прогон + классификация + опц. torch |

## `Doc/` — документация

| Путь | Назначение |
|------|------------|
| `Doc/architecture/` | модель C4 (C1–C4) с Mermaid-диаграммами |
| `Doc/folders.md` | этот файл — карта каталогов |
| `Doc/classes.md` | каталог классов с ответственностью |

## `MemoryBank/` — рабочая память ассистента

| Путь | Назначение |
|------|------------|
| `MemoryBank/MASTER_INDEX.md` | карта состояния проекта (читать первым) |
| `MemoryBank/tasks/` | активные задачи (`IN_PROGRESS.md`, `TASK_*`) |
| `MemoryBank/sessions/` | журнал сессий `YYYY-MM-DD.md` |
| `MemoryBank/changelog/` | помесячный changelog |
| `MemoryBank/specs/` | спеки, ревью, исследования |

## `.claude/` — конфигурация Claude Code

| Путь | Назначение |
|------|------------|
| `.claude/rules/` | 6 правил (workflow, профиль, NO-pytest, стиль, worktree-safety) |
| `.claude/hooks/` | bash-хуки (защита команд, напоминания) |
| `.claude/settings.json` | permissions + wiring хуков |

## Прочее (вне версионного контроля или справочное)

| Путь | Назначение |
|------|------------|
| `out/` | **артефакты прогона** (figures `*.png`, data `*.npy`) — в `.gitignore` |
| `Example/` | референсные прототипы и графики (radar_stft, cube_*, …) |
| `Charts/` | вспомогательные материалы |
| `.venv/` | виртуальное окружение Python 3.12 (torch-ROCm) — в `.gitignore` |
