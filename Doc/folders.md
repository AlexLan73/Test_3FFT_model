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

Апертура **i×j** (не обязательно квадрат): каждая ось паддится до ближайшей
степени двойки — `ArrayConfig.padded_shape()`. Полный когнитивный конвейер:
фронтенд (2FFT/3DFFT) → куб → токенизатор → арбитр → целеуказание → трекинг
(+ ветка anti-barrage). См. таблицу «Конвейер» в [`README.md`](../README.md).

| Подпакет | Назначение | Ключевые файлы |
|----------|------------|----------------|
| `core/config/` | неизменяемые параметры прогона (Value Objects), i×j | `array_config.py` (`ArrayConfig.padded_shape`), `scene_config.py`, `simulation_config.py`, `project_config.py` |
| `core/generators/` | синтез сырого куба из сцены, такты, объём | `sources.py`, `jammers.py`, `scene.py`, `factory.py`, `grid.py`, `scene_modeler.py`, `tact_sequence.py`, `volume.py` |
| `core/generators/waveforms/` | генерация сигналов (сырое время → куб) | `cw.py`/`lfm.py`/`am.py`/`phase_code.py`/`fm.py`, `jammers_rf.py`, `mseq.py`, `waveform_to_cube.py` (`LfmToCube`/`AmToCube`), `factory.py` |
| `core/generators/backends/` | бэкенд генерации (Strategy) | `base.py` (`GenBackend`), `numpy_backend.py` |
| `core/models/` | спектральное преобразование: 2 раздельных FFT (дальностный + угловой) | `angular_fft.py`, `range_fft.py`, `fft3d.py`, `base.py`, `windows.py`, `result.py` |
| `core/models/tokenizer/` | объёмный токенизатор (признаки, триаж, арбитр, гл.4–5) | `cfar.py` (`OsCfarDetector`), `features.py`, `triage.py`, `tokenizer.py`, `arbiter.py` (гл.5), `tokens.py`, `calibration.py` |
| `core/models/targeting/` | целеуказание пучка FM-m (гл.8) | `beam.py` (`BeamTargeting`), `cycle.py` (`CognitiveCycle`), `roi_gate.py` |
| `core/models/tracking/` | трекинг детекций между тактами | `track.py`, `tracker.py` (`NearestNeighborTracker`) |
| `core/models/anti_barrage/` | подавление заградительной помехи + CFAR/кластеризация | `nuller.py` (`SubspaceNuller`), `mvdr.py` (`RobustMvdrNuller`), `pipeline.py` (`AntiBarragePipeline`), `cfar.py` (`CaCfarDetector`), `clustering.py` |
| `core/models/classification/` | классификация куба | `classifier.py`, `rule_based.py`, `cnn3d.py`, `dataset.py`, `labels.py` |
| `core/graphics/` | визуализаторы (matplotlib) + запись фигур | `visualizer.py`, `cube_scatter.py`, `angular_map.py`, `range_profile.py`, `writer.py`, `layout.py`, `sampling.py`, `square_view.py` |
| `core/graphics/interactive/` | интерактивная ветка (plotly, опц.) | `interactive_visualizer.py`, `cube_interactive.py`, `html_writer.py` |
| `core/graphics/panel/` | живая панель управления сценой (Observer, Dear PyGui опц.) | `panel_model.py` (GUI-free), `panel_app.py` |
| `core/data_context/` | хранение кубов (Facade + Repository) + шина сообщений | `data_context.py`, `repository.py`, `message_bus.py` (`MessageBus`/`Observer`), `run_workspace.py` |
| `core/runtime/` | межпроцессный транспорт панели (ZMQ + msgpack) | `transport.py` (`Transport`/`FanOutTransport`), `codec.py`, `scene_server.py` (`SceneServer`), `commands.py` |
| `core/motion/` | кинематика движения цели | `state.py` (`TargetState`), `models.py` (`MotionModel` + варианты), `kinematics.py` |
| `core/snr/` | SNR-эстиматоры (спектр CA/OS-CFAR + статистика) | `estimator.py`, `config.py`, `signal.py` |
| `core/gpu_libs/` | loader GPU `.so` (копии из DSP-GPU, cp313) | `loader.py`, `configGPU.json` |
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
