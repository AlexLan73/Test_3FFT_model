# 🧱 Каталог классов radar3d

Все публичные классы с ответственностью, ключевыми методами и применённым
паттерном. Сверено с исходниками `core/`. Группировка — по подпакетам.

---

## `core.config` — параметры прогона (Value Objects)

| Класс | Ответственность | Поля / методы |
|-------|-----------------|---------------|
| `ArrayConfig` | размер квадратной решётки | `nx=16`, `ny=16`; валидация в `__post_init__` |
| `RangeConfig` | дальностная (быстрая) ось | `n_real`, `n_fft`; `is_zero_padded`; `n_fft ≥ n_real` |
| `SimulationConfig` | корневой конфиг прогона | `array`, `range`, `scene`, `seed=7` |
| `EmitterSpec` | базовая спецификация излучателя | `kx`, `ky`, `amplitude` |
| `TargetSpec` | истинная точечная цель | `+ range_bin`, `phase` |
| `DrfmCombSpec` | гребёнка ложных целей DRFM | `+ lead_bin`, `spacing`, `count`, `decay` |
| `BarrageSpec` | заградительная шумовая помеха | `+ power` |
| `HamEmitterSpec` | стороннее излучение (радиолюбитель) | `+ chirp_rate` |
| `ThermalNoiseSpec` | тепловой шум приёмника | `power` |
| `SceneConfig` | набор спецификаций сцены | `emitters`, `thermal` |
| `default_scenario()` | фабрика эталонной сцены | → `SimulationConfig` |

> Все `@dataclass(frozen=True)` — неизменяемы, безопасны для шаринга.

---

## `core.generators` — синтез сырого куба

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|------------------|
| `SignalSource` (ABC) | вклад источника в куб | Strategy; `contribute(grid, rng, rs)`, `_empty()` |
| `_SteeredTone` | базовый наведённый тон | `_tone()`, `_steer(grid)` |
| `PointTarget` | точечная цель на дальности | наследник `_SteeredTone` |
| `DrfmComb` | гребёнка копий **позади** фронта | наследник `_SteeredTone` |
| `BarrageJammer` | заливка дальности с направления | наследник `SignalSource` |
| `HamEmitter` | размаз по дальности после дерампа | наследник `SignalSource` |
| `ThermalNoise` | ненаправленный шум по элементам | наследник `SignalSource` |
| `Scene` | композит источников (Σ вкладов) | **Composite**; `add()`, `contribute()` |
| `SceneBuilder` | строит `Scene` из `SceneConfig` | **Builder**; `build(cfg)` |
| `EmitterFactory` | спека → объект-источник | **Abstract Factory + Registry**; `register()`, `create()` |
| `Synthesizer` | сырой куб `(nx,ny,n_real)` из сцены | Pure Fabrication; `build(scene)` |
| `ArrayGrid` | фазовый вектор наведения | Information Expert; `from_config()`, `steering(kx,ky)` |

---

## `core.models` — спектральное преобразование

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|------------------|
| `RadarModel` (ABC) | сырой куб → `SpectralCube` | **Template Method** `process()`; абстрактные `_apply_windows/_transform/_build_result` |
| `Fft3DModel` | 3D-БПФ (углы центрируются, дальность односторонняя) | `np.fftn` + `fftshift` по угловым осям |
| `WindowFunction` (ABC) | весовое окно по оси | **Strategy**; `taper(n)` |
| `RectWindow` / `HannWindow` / `HammingWindow` | конкретные окна | — |
| `AxisWindows` | тройка окон по 3 осям куба | OCP; `apply(cube)` |
| `SpectralCube` | спектральный куб \|C\| + оси | Information Expert; `magnitude(_db)`, `index_of_angle()`, `angular_energy_db()`, `range_profile_db()` |
| `Axis` | описание одной оси | `name`, `values`, `centered` |

---

## `core.models.classification` — классификация

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|------------------|
| `CubeClassifier` (ABC) | отнести куб к классу | **Strategy**; `classify(cube) → Classification` |
| `RuleBasedClassifier` | детерминированный различитель формы отклика | без torch; `_count_peaks()`, `_make()` |
| `Cnn3DClassifier` | обучаемая 3D-CNN (инференс) | LSP-замена RuleBased; torch ленивый импорт |
| `build_cnn3d()` | сборка сети 2×Conv3d→GAP→FC (~4К парам.) | функция-фабрика |
| `CubeDatasetGenerator` | размеченные кубы из того же генератора | `_scene_for()`, `sample()`, `batch()` |
| `Classification` | решение классификатора | `label`, `name`, `confidence`, `probabilities`, `cell`; `__str__` |
| `CLASS_NAMES` | таксономия | `("empty","target","barrage","comb","ham")` |

---

## `core.graphics` — визуализация

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|------------------|
| `Visualizer` (ABC) | куб → matplotlib `Figure` | **Strategy**; `render(cube)` |
| `CubeScatterVisualizer` | 3D-скаттер ячеек выше порога | `threshold_db`, `range_limit` |
| `AngularMapVisualizer` | угловая карта энергии в гейте | `gate_kx/ky`, `gate_half` |
| `RangeProfileVisualizer` | дальностные профили выбранных ячеек | `cells`, `range_limit` |
| `FigureWriter` | запись `Figure` в PNG | Pure Fabrication; `write(fig, name)` |

---

## `core.data_context` — хранение

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|------------------|
| `DataContext` | единая точка load/save | **Facade**; `save_cube()`, `load_cube()` |
| `CubeRepository` (ABC) | контракт хранилища куба | Repository; `save()`, `load()` |
| `NpyCubeRepository` | хранение в `.npy` | `_path()`, `save()`, `load()` |

---

## `core.controller` — координация

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|------------------|
| `SimulationController` | оркестрация прогона | **GRASP Controller**, DIP; `run(cfg, save_as)` |
| `ProcessingOutcome` | результат прогона | `raw_cube`, `spectral_cube` |

---

## `common` — тест-инфраструктура

| Класс | Ответственность | Методы |
|-------|-----------------|--------|
| `TestRunner` | базовый раннер (замена pytest) | `setup()`, `run_all()` |
| `AssertionGroup` | копит проверки одного теста | `add(cond, msg)`, `ok` |
| `SkipTest` | пропуск теста (исключение) | — |

→ См. связи на диаграмме [C4 — Code](architecture/C4-code.md).
