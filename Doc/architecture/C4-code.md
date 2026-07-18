# C4 — Code (классы и связи)

> Самый детальный уровень: иерархии классов и их отношения (UML). Сгруппировано
> по подсистемам когнитивного конвейера. Сверено с исходниками `core/`.
> Полное описание классов — [`Doc/classes.md`](../classes.md).

## Источники сигналов и волны (Composite + Strategy)

```mermaid
classDiagram
    class SignalSource {
        <<abstract>>
        +contribute(grid, rng, rs) ndarray
    }
    class Scene {
        +add(source) Scene
        +contribute(grid, rng, rs)
    }
    class Waveform {
        <<abstract>>
    }
    class WaveformFactory {
        +register(spec_type, builder)
        +create(spec) Waveform
    }
    class WaveformToCube {
        <<Protocol>>
        +fill(volume, cfg) SpectralCube
    }
    class LfmToCube {
        +fill(volume, cfg) SpectralCube
    }
    class AmToCube {
        +fill(volume, cfg) SpectralCube
        +scan(volume, cfg) list
    }

    SignalSource <|-- Scene
    Scene o-- "0..*" SignalSource : композит
    Waveform <|-- CwWaveform
    Waveform <|-- LfmWaveform
    Waveform <|-- AmWaveform
    Waveform <|-- PhaseCodeWaveform
    Waveform <|-- FmInterferenceWaveform
    WaveformFactory ..> Waveform : создаёт
    WaveformToCube <|.. LfmToCube
    WaveformToCube <|.. AmToCube
    LfmToCube ..> SpectralCube : создаёт (2 FFT, точно)
    AmToCube ..> SpectralCube : создаёт (скольз. 3D-FFT, грубо)
```

## Модель + окна + результат (i×j / 2ⁿ)

```mermaid
classDiagram
    class RadarModel {
        <<abstract>>
        +process(datacube) SpectralCube
        #_apply_windows(cube)*
        #_transform(cube)*
        #_build_result(spectrum)*
    }
    class Fft3DModel {
        -_array: ArrayConfig
        -_rng: RangeConfig
        -_windows: AxisWindows
    }
    class ArrayConfig {
        +nx; +ny
        +padded_shape() tuple
    }
    class WindowFunction {
        <<abstract>>
        +taper(n) ndarray
    }
    class AxisWindows { +apply(cube) ndarray }
    class SpectralCube {
        +magnitude ndarray
        +magnitude_db ndarray
        +index_of_angle(kx,ky)
        +angular_energy_db()
        +range_profile_db(ix,iy)
    }
    class Axis { +name; +values; +centered }

    RadarModel <|-- Fft3DModel
    Fft3DModel o-- ArrayConfig
    WindowFunction <|-- RectWindow
    WindowFunction <|-- HannWindow
    WindowFunction <|-- HammingWindow
    Fft3DModel o-- AxisWindows
    AxisWindows o-- "3" WindowFunction
    Fft3DModel ..> SpectralCube : создаёт
    SpectralCube o-- "3" Axis
    note for ArrayConfig "angular_fft() паддит апертуру\nдо N_pad=2ⁿ на каждой оси\nsinθ = k/(N_pad/2)"
```

## Токенизатор (гл.4/4-бис, Template Method) + Арбитр (гл.5, Composite)

```mermaid
classDiagram
    class OsCfarDetector {
        +detect_mask(power) ndarray
        +find_peaks(power) list
    }
    class FeatureExtractor { +extract(power) FeatureVector }
    class FeatureVector { <<VO>> }
    class SliceTriage {
        <<abstract>>
        +classify(f) tuple
    }
    class RuleBasedTriage
    class VolumeTokenizer {
        +tokenize(cube) list~SliceToken~
    }
    class SliceToken { +n_peaks }
    class RangeVerdict { <<VO>> }
    class Arbiter {
        <<abstract>>
        +arbitrate(verdicts) list~TargetDecision~
    }
    class EdgeArbiter { +arbitrate(verdicts) }
    class CodeArbiter { +arbitrate(verdicts) }
    class CombinedArbiter { +arbitrate(verdicts) }
    class TargetDecision { <<VO>> }

    SliceTriage <|-- RuleBasedTriage
    VolumeTokenizer o-- OsCfarDetector
    VolumeTokenizer o-- FeatureExtractor
    VolumeTokenizer o-- SliceTriage
    VolumeTokenizer ..> SliceToken : создаёт
    VolumeTokenizer ..> RangeVerdict : assemble_range()
    Arbiter <|-- EdgeArbiter
    Arbiter <|-- CodeArbiter
    Arbiter <|-- CombinedArbiter
    CombinedArbiter o-- EdgeArbiter : композит
    CombinedArbiter o-- CodeArbiter : композит
    Arbiter ..> TargetDecision : возвращает
    RangeVerdict --> Arbiter : вход arbitrate()
```

## Целеуказание (гл.8, Facade) + Трекинг

```mermaid
classDiagram
    class CognitiveCycle {
        +step(cube) CycleResult
    }
    class CycleResult { <<VO>> }
    class Targeting {
        <<abstract>>
        +point(decisions) list~BeamCommand~
    }
    class BeamTargeting { +point(decisions) }
    class BeamCommand { <<VO>> }
    class RoiGate { +filter(detections, beams) }
    class Tracker {
        <<abstract>>
        +update(decisions, tact) list~Track~
    }
    class NearestNeighborTracker { +update(decisions, tact) }
    class Track { <<VO>> }

    CognitiveCycle o-- VolumeTokenizer
    CognitiveCycle o-- Arbiter
    CognitiveCycle o-- Targeting
    CognitiveCycle ..> CycleResult : создаёт
    Targeting <|-- BeamTargeting
    Targeting ..> BeamCommand : возвращает
    RoiGate ..> BeamCommand : использует
    Tracker <|-- NearestNeighborTracker
    Tracker ..> Track : возвращает
```

## Anti-barrage (Facade)

```mermaid
classDiagram
    class SubspaceNuller {
        +decompose(datacube)
        +apply(datacube) ndarray
        +report(datacube) NullerReport
    }
    class RobustMvdrNuller {
        +weights(datacube) ndarray
        +apply(datacube) ndarray
    }
    class NullerReport { <<VO>> }
    class CaCfarDetector { +detect(cube) list~Detection~ }
    class Detection { <<VO>> }
    class DetectionClusterer { +cluster(detections) list~DetectionCluster~ }
    class DetectionCluster { <<VO>> }
    class AntiBarragePipeline { +process(datacube) list~Detection~ }

    SubspaceNuller ..> NullerReport : создаёт
    AntiBarragePipeline o-- SubspaceNuller
    AntiBarragePipeline o-- RobustMvdrNuller
    AntiBarragePipeline o-- CaCfarDetector
    AntiBarragePipeline o-- DetectionClusterer
    CaCfarDetector ..> Detection : создаёт
    DetectionClusterer ..> DetectionCluster : создаёт
```

## Классификация (Strategy + LSP)

```mermaid
classDiagram
    class CubeClassifier {
        <<abstract>>
        +classify(cube) Classification
    }
    class RuleBasedClassifier { +classify(cube) Classification }
    class Cnn3DClassifier {
        -_net; -_device
        +classify(cube) Classification
    }
    class Classification {
        +label: int
        +name: str
        +confidence: float
        +probabilities: dict
        +cell: tuple
    }
    class CubeDatasetGenerator {
        +sample(name)
        +batch(n, balanced)
    }
    CubeClassifier <|-- RuleBasedClassifier
    CubeClassifier <|-- Cnn3DClassifier
    CubeClassifier ..> Classification : возвращает
    Cnn3DClassifier ..> CubeDatasetGenerator : обучается на
```

## Runtime (панель, Observer/Composite) + Motion

```mermaid
classDiagram
    class Transport {
        <<abstract>>
        +publish(topic, tact, payload)
    }
    class ZmqTransport
    class WebSocketTransport
    class FanOutTransport {
        +publish(topic, tact, payload)
    }
    class SceneServer {
        +step() tuple
        +run(n_tacts)
    }
    class Command {
        <<abstract>>
    }
    class MessageBus {
        +subscribe(observer)
        +publish(event)
    }
    class MotionModel {
        <<Protocol>>
        +propagate(state, dt, rng) TargetState
    }
    class TargetState { <<VO>> }

    Transport <|-- ZmqTransport
    Transport <|-- WebSocketTransport
    Transport <|-- FanOutTransport
    FanOutTransport o-- "0..*" Transport : композит
    SceneServer o-- Transport
    Command <|-- AddTarget
    Command <|-- RemoveTarget
    Command <|-- SetMotion
    Command <|-- EnableJammer
    MotionModel <|.. ConstantVelocity
    MotionModel <|.. MarkovDrift
    MotionModel <|.. CoordinatedTurn
    MotionModel <|.. ConstantAccel
    MotionModel <|.. WeavingManeuver
    MotionModel ..> TargetState : propagate()
```

## Координация, хранение, графика

```mermaid
classDiagram
    class SimulationController { +run(cfg, save_as) ProcessingOutcome }
    class SceneBuilder { +build(cfg) Scene }
    class EmitterFactory {
        +register(spec_type, builder)
        +create(spec) SignalSource
    }
    class DataContext {
        +save_cube(name, cube)
        +load_cube(name)
    }
    class CubeRepository { <<abstract>> }
    class NpyCubeRepository
    class Visualizer { <<abstract>> +render(cube) Figure }
    class FigureWriter { +write(fig, name) str }

    SimulationController o-- RadarModel
    SimulationController o-- SceneBuilder
    SimulationController o-- DataContext
    SceneBuilder o-- EmitterFactory
    DataContext o-- CubeRepository
    CubeRepository <|-- NpyCubeRepository
    Visualizer <|-- CubeScatterVisualizer
    Visualizer <|-- AngularMapVisualizer
    Visualizer <|-- RangeProfileVisualizer
```

## Применённые паттерны (GoF / GRASP)

| Паттерн | Где |
|---------|-----|
| **Strategy** | `WindowFunction`, `RadarModel`, `Visualizer`/`InteractiveVisualizer`, `CubeClassifier`, `WaveformToCube` (`LfmToCube`/`AmToCube`), `SliceTriage`, `Arbiter` (`EdgeArbiter`/`CodeArbiter`), `Targeting` (`BeamTargeting`), `Tracker` (`NearestNeighborTracker`), `Transport`, `MotionModel` |
| **Composite** | `Scene` из `SignalSource`; `CombinedArbiter` из `EdgeArbiter`+`CodeArbiter`; `FanOutTransport` из `Transport` |
| **Facade** | `DataContext`, `CognitiveCycle` (токенизатор+арбитр+целеуказание), `AntiBarragePipeline` (nuller+CFAR+кластеризация) |
| **Abstract Factory + Registry** | `EmitterFactory`, `WaveformFactory` |
| **Builder** | `SceneBuilder` |
| **Template Method** | `RadarModel.process`, `VolumeTokenizer.tokenize` |
| **Value Object** | конфиги (`ArrayConfig`/`ProjectConfig`), `SpectralCube`, `Axis`, `Classification`, `SliceToken`/`RangeVerdict`, `TargetDecision`, `BeamCommand`, `Track`, `Detection`, `NullerReport`, `TargetState` |
| **Observer** | `MessageBus` (внутрипроцессный), `SceneServer`/`Transport` (межпроцессный) |
| **Command** | `core.runtime.commands` (`AddTarget`/`RemoveTarget`/`SetMotion`/`EnableJammer`/`Step`) |
| **Protocol / DIP** | `WaveformToCube`, `SnrEstimator`, `MotionModel`, `GenBackend` |
| **Pure Fabrication** | `FigureWriter`, `Synthesizer`, `VolumeBuilder`, репозитории |
| **Dependency Injection** | связывание в `main.py`/`demo_*.py` (Composition Root) |
| **Information Expert** | `SpectralCube` (выборки), `ArrayGrid` (фаза наведения), `ArrayConfig.padded_shape()` |

→ Назад: [C3](C3-component.md) · [Каталог классов](../classes.md)
