# 🧱 Каталог классов radar3d

Все публичные классы с ответственностью, ключевыми методами и применённым
паттерном. Сверено с исходниками `core/`. Группировка — по подпакетам.

---

## `core.config` — параметры прогона (Value Objects)

| Класс | Ответственность | Поля / методы |
|-------|-----------------|---------------|
| `ArrayConfig` | апертура **i×j** (`nx≠ny` допустим) | `nx`, `ny`; `padded_shape()` → паддинг до 2ⁿ по каждой оси независимо |
| `RangeConfig` | дальностная (быстрая) ось | `n_real`, `n_fft`; `is_zero_padded`; `n_fft ≥ n_real` |
| `ProjectConfig` | корневой конфиг такта (P1+) | объединяет `ArrayConfig`/`RangeConfig`/волну/сцену |
| `SimulationConfig` | корневой конфиг прогона (main.py-тракт) | `array`, `range`, `scene`, `seed=7` |
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
| `SceneModeler` | помехи сцены → вклад в общий куб такта | `build_jammers(cfg)`, `contribute_to(volume, cfg, ...)` |
| `VolumeBuilder` | куб такта из состояния цели + сцены | `build()`, `build_from_sample()`, `add_shared_noise()` |
| `Tact`/`TactSequence` | один такт цели / итератор тактов (одна цель) | `Iterator[Tact]` |
| `MultiTact`/`MultiTactSequence`/`TargetHandle` | такт/итератор для **нескольких** целей | `Iterator[MultiTact]` |
| `iter_cubes`/`iter_multi_cubes` | тактовая последовательность → кубы | генераторные функции |
| `GenBackend` (ABC) | бэкенд генерации сырого времени | **Strategy** (`core.generators.backends`); `NumpyBackend` — реализация |

---

## `core.models` — спектральное преобразование (i×j / 2ⁿ)

| Класс / функция | Ответственность | Паттерн / методы |
|-------|-----------------|------------------|
| `RadarModel` (ABC) | сырой куб → `SpectralCube` | **Template Method** `process()`; абстрактные `_apply_windows/_transform/_build_result` |
| `Fft3DModel` | 3D-БПФ / скользящий (углы центрируются, дальность односторонняя) | `_apply_windows`/`_transform`/`_build_result` |
| `angular_fft(cube, aperture_window=None)` | угловой FFT поячеечно, паддинг апертуры до 2ⁿ, `sinθ=k/(N_pad/2)` | функция; точный тракт ЛЧМ |
| `RangeFft` | дальностный FFT (глобальный, отдельно от углового) | `n_fft_for(n)`, `transform(dechirped, fs, mu)`; `next_pow2`, `k_signed_range_axis` |
| `WindowFunction` (ABC) | весовое окно по оси | **Strategy**; `taper(n)` |
| `RectWindow` / `HannWindow` / `HammingWindow` | конкретные окна | — |
| `AxisWindows` | тройка окон по 3 осям куба | OCP; `apply(cube)` |
| `SpectralCube` | спектральный куб \|C\| + оси | Information Expert; `magnitude(_db)`, `index_of_angle()`, `angular_energy_db()`, `range_profile_db()` |
| `Axis` | описание одной оси | `name`, `values`, `centered` |

### `core.generators.waveforms` — фронтенд (Strategy: точно/грубо)

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|-------------------|
| `WaveformToCube` (Protocol) | сигнал → заполнение куба | **Strategy**; `fill(volume, cfg) -> SpectralCube` |
| `LfmToCube` | ЛЧМ: 2 раздельных FFT (точный тракт) | `fill()`; `build_lfm_target_volume()` — фикс инъекции цели (A9-gap1) |
| `AmToCube` | АМ: скользящий 3D-FFT по окну nx×ny×D (грубый тракт) | `fill()`, `scan()` — список кубов по окнам |
| `CwWaveform`/`LfmWaveform`/`AmWaveform`/`PhaseCodeWaveform`/`FmInterferenceWaveform` | конкретные волны (P1/P4) | наследники `Waveform` |
| `WaveformFactory` | спека → волна | **Abstract Factory + Registry** |
| `BarrageRfJammer`/`SmspJammer`/`DrfmRepeaterJammer`/`IndustrialCwJammer`/`ImpulsiveArcJammer`/`VfdHarmonicJammer` | помехи РФ (P5, `jammers_rf.py`) | наследники `Waveform` |
| `m_sequence`/`m_sequence_pow2` | М-последовательность (код FM-m) | функции (`mseq.py`) |

---

## `core.models.tokenizer` — токенизатор (гл.4 + гл.4-бис) и арбитр (гл.5)

| Класс / функция | Ответственность | Паттерн / методы |
|-------|-----------------|-------------------|
| `OsCfarDetector` | точный OS-CFAR (Pfa по формуле Rohling) | `detect_mask()`, `find_peaks()`, `cell_threshold()` |
| `FeatureExtractor` | 6 признаков слайса, норм. на M=N_pad_x·N_pad_y | `extract(power) -> FeatureVector` |
| `FeatureVector` | вектор признаков слайса (VO) | — |
| `SliceTriage` (ABC) | проход 1: слайс → {noise/source/smeared} | **Strategy**; `classify(f) -> (label, score)` |
| `RuleBasedTriage` | детерминированный триаж по признакам | реализация `SliceTriage` |
| `NOISE`/`SOURCE`/`SMEARED` | метки прохода 1 | константы |
| `VolumeTokenizer` | обход куба, поиск пиков, сборка токенов | **Template Method**; `tokenize(cube) -> list[SliceToken]` |
| `assemble_range(...)` | проход 2: по дальности → {target/comb/barrage} | функция; `_dominant_period`, `_classify_group` |
| `SliceToken`/`PeakInfo` | токен слайса (пики + метка) (VO) | `n_peaks` |
| `RangeVerdict` | вердикт прохода 2 по дальностной группе (VO) | — |
| `TARGET`/`COMB`/`BARRAGE` | метки прохода 2 | константы |
| `Arbiter` (ABC) | вердикты → финальные решения | **Strategy**; `arbitrate(verdicts) -> list[TargetDecision]` |
| `EdgeArbiter` | передний край τ≥0 (геометрия), чистит осколки/джиттер заграда | реализация `Arbiter` |
| `CodeArbiter` | свежесть FM-m кода (корреляция) | реализация `Arbiter`; `fm_correlate(ref, inp)` |
| `CombinedArbiter` | объединяет edge+code | **Composite** над `Arbiter`; `arbitrate()` |
| `TargetDecision` | финальное решение арбитра (VO) | — |
| `TriageCalibrator` | калибровка/валидация триажа по апертурам и SNR | `build_dataset()`, `validate()`, `class_stats()` |

---

## `core.models.targeting` — целеуказание пучка FM-m (гл.8)

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|-------------------|
| `Targeting` (ABC) | решения → команды на пучок лучей | **Strategy**; `point(decisions) -> list[BeamCommand]` |
| `BeamTargeting` | пучок лучей в конус неопределённости вокруг цели | реализация `Targeting`; `_cone_beam_angles()` |
| `BeamCommand` | команда наведения луча (VO) | — |
| `CognitiveCycle` | один когнитивный такт: токенизация→арбитраж→целеуказание | **Facade**; `step(cube) -> CycleResult` |
| `CycleResult` | результат такта (VO) | — |
| `RoiGate` | фильтр детекций по ROI активных лучей | `filter(detections, beams)` |

---

## `core.models.tracking` — трекинг между тактами

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|-------------------|
| `Tracker` (ABC) | решения такта → обновлённые треки | **Strategy**; `update(decisions, tact) -> list[Track]` |
| `NearestNeighborTracker` | ассоциация по ближайшему соседу + линейная регрессия скорости | реализация `Tracker`; `_to_track()` |
| `Track` | трек цели между тактами (VO) | — |

---

## `core.models.anti_barrage` — подавление заградительной помехи

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|-------------------|
| `SubspaceNuller` | проекция на ортогональное подпространство помехи (eigh) | `decompose()`, `apply()`, `report() -> NullerReport` |
| `NullerReport` | отчёт о подавлении (VO) | — |
| `RobustMvdrNuller` | Capon-веса + диагональная нагрузка (робастный MVDR) | `weights()`, `apply()` |
| `CaCfarDetector` | CA-CFAR по кубу | `detect(cube) -> list[Detection]`, `alpha`/`pfa`/`n_train`/`n_guard` |
| `Detection` | обнаружение CFAR (VO) | — |
| `DetectionClusterer` | кластеризация детекций по углу/дальности | `cluster(detections) -> list[DetectionCluster]` |
| `DetectionCluster` | кластер детекций (VO) | — |
| `AntiBarragePipeline` | nuller → CFAR → кластеризация | **Facade**; `process(datacube) -> list[Detection]` |

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
| `AxisLayout` | раскладка осей куба на экран (VO) | `range_vertical()`, `range_in_depth()` |
| `CubeSampler` | порог + выборка точек по глобальной `magnitude_db` | `mask = db > threshold`, срез по `range_limit` |
| `SquareView`/`SquareToken` | контрольный вид reduce+argmax по кубу | Pure Fabrication |

### `core.graphics.interactive` (опц., plotly) и `core.graphics.panel` (опц., Dear PyGui)

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|-------------------|
| `InteractiveVisualizer` (ABC) | куб → plotly `Figure` | **Strategy**; параллельна `Visualizer` |
| `InteractiveCubeVisualizer` | 3D-скаттер plotly | реализация `InteractiveVisualizer` |
| `HtmlWriter` | запись plotly-фигуры в HTML | Pure Fabrication |
| `PanelModel` | чистая дата-модель живой панели (без dearpygui) | `Field`/`Cell`/`Element`/`SignalBlock`, `lerp_field()` |

---

## `core.data_context` — хранение + внутрипроцессная шина

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|------------------|
| `DataContext` | единая точка load/save | **Facade**; `save_cube()`, `load_cube()` |
| `CubeRepository` (ABC) | контракт хранилища куба | Repository; `save()`, `load()` |
| `NpyCubeRepository` | хранение в `.npy` | `_path()`, `save()`, `load()` |
| `MessageBus` | внутрипроцессная шина событий (Observer) | `subscribe()`, `publish()` |
| `Observer` (ABC) | подписчик на события шины | **Strategy**/Observer |
| `RunWorkspace` | рабочая директория прогона (конфиг ↔ yaml) | `config_to_dict`/`to_yaml`/`from_yaml` |

---

## `core.runtime` — межпроцессный транспорт панели (ZMQ + msgpack)

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|-------------------|
| `Transport` (ABC) | контракт транспорта между процессами | **Strategy**; `ZmqTransport`/`WebSocketTransport` — реализации |
| `FanOutTransport` | рассылка одного сообщения нескольким транспортам | **Composite** над `Transport` |
| `SceneServer` | издатель такта сцены + приём команд панели | `step()`, `run(n_tacts)`, `_tracks_payload()` |
| `SceneState`/`LiveTarget` | состояние сцены на сервере (VO) | — |
| `Command` (ABC) | команда панель→сервер | **Command pattern**; `AddTarget`/`RemoveTarget`/`SetMotion`/`EnableJammer`/`Step`/`SetNeighborPlanes` |
| `codec` | msgpack-кодек сообщений | модуль-функции |

---

## `core.motion` — кинематика цели

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|-------------------|
| `TargetState` | состояние цели (позиция/скорость) (VO) | — |
| `MotionModel` (Protocol) | закон движения | **Strategy**; `propagate(state, dt, rng) -> TargetState` |
| `ConstantVelocity`/`MarkovDrift`/`CoordinatedTurn`/`ConstantAccel`/`WeavingManeuver` | конкретные законы движения | реализации `MotionModel` |
| `Kinematics`/`KinematicsSample` | проекция движения в бины (дальность/угол) | — |

---

## `core.snr` — оценка SNR

| Класс | Ответственность | Паттерн / методы |
|-------|-----------------|-------------------|
| `SnrEstimator` (Protocol) | оценка SNR по временно́му ряду | **Strategy**; `estimate(...) -> SnrResult` |
| `SpectrumSnrEstimator` | CA/OS-CFAR по FFT (порт GPUWorkLib) | реализация `SnrEstimator` |
| `StatisticsSnrEstimator` | time-domain статистика | реализация `SnrEstimator` |
| `SnrConfig`/`SnrResult` | конфиг/результат оценки (VO) | — |
| `PointSignalGenerator` | генератор строб-тона + AWGN | `generate()` |

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
