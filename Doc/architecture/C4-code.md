# C4 — Code (классы и связи)

> Самый детальный уровень: иерархии классов и их отношения (UML). Сгруппировано
> по подсистемам. Сверено с исходниками `core/`.

## Источники сигналов (Composite + наследование)

```mermaid
classDiagram
    class SignalSource {
        <<abstract>>
        +contribute(grid, rng, rs) ndarray
        #_empty(grid, rng) ndarray
    }
    class Scene {
        -_sources: list
        +add(source) Scene
        +contribute(grid, rng, rs)
    }
    class _SteeredTone {
        +kx: float
        +ky: float
        #_tone(...) ndarray
        #_steer(grid) ndarray
    }
    class PointTarget
    class DrfmComb
    class BarrageJammer
    class HamEmitter
    class ThermalNoise

    SignalSource <|-- Scene
    SignalSource <|-- _SteeredTone
    SignalSource <|-- BarrageJammer
    SignalSource <|-- HamEmitter
    SignalSource <|-- ThermalNoise
    _SteeredTone <|-- PointTarget
    _SteeredTone <|-- DrfmComb
    Scene o-- "0..*" SignalSource : композит
```

## Спецификации сцены (Value Objects)

```mermaid
classDiagram
    class EmitterSpec {
        +kx: float
        +ky: float
        +amplitude: float
    }
    class TargetSpec {
        +range_bin: float
        +phase: float
    }
    class DrfmCombSpec {
        +lead_bin, spacing, count, decay
    }
    class BarrageSpec { +power }
    class HamEmitterSpec { +chirp_rate }
    class SceneConfig {
        +emitters: Sequence
        +thermal: ThermalNoiseSpec
    }
    EmitterSpec <|-- TargetSpec
    EmitterSpec <|-- DrfmCombSpec
    EmitterSpec <|-- BarrageSpec
    EmitterSpec <|-- HamEmitterSpec
    SceneConfig o-- "0..*" EmitterSpec
    SceneConfig o-- ThermalNoiseSpec
```

## Модель + окна + результат

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
    class WindowFunction {
        <<abstract>>
        +taper(n) ndarray
    }
    class AxisWindows {
        +apply(cube) ndarray
    }
    class SpectralCube {
        +magnitude ndarray
        +magnitude_db ndarray
        +index_of_angle(kx,ky)
        +angular_energy_db()
        +range_profile_db(ix,iy)
    }
    class Axis { +name; +values; +centered }

    RadarModel <|-- Fft3DModel
    WindowFunction <|-- RectWindow
    WindowFunction <|-- HannWindow
    WindowFunction <|-- HammingWindow
    Fft3DModel o-- AxisWindows
    AxisWindows o-- "3" WindowFunction
    Fft3DModel ..> SpectralCube : создаёт
    SpectralCube o-- "3" Axis
```

## Классификация (Strategy + LSP)

```mermaid
classDiagram
    class CubeClassifier {
        <<abstract>>
        +classify(cube) Classification
    }
    class RuleBasedClassifier {
        +classify(cube) Classification
    }
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

## Координация, хранение, графика

```mermaid
classDiagram
    class SimulationController {
        +run(cfg, save_as) ProcessingOutcome
    }
    class SceneBuilder { +build(cfg) Scene }
    class EmitterFactory {
        +register(spec_type, builder)
        +create(spec) SignalSource
    }
    class Synthesizer { +build(scene) ndarray }
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
| **Strategy** | `WindowFunction`, `RadarModel`, `Visualizer`, `CubeClassifier` |
| **Composite** | `Scene` из `SignalSource` |
| **Abstract Factory + Registry** | `EmitterFactory` (спека → источник, OCP) |
| **Builder** | `SceneBuilder` |
| **Template Method** | `RadarModel.process` (окно → преобразование → упаковка) |
| **Facade** | `DataContext` над репозиториями |
| **Value Object** | конфиги, `*Spec`, `SpectralCube`, `Axis`, `Classification` |
| **Pure Fabrication** | `FigureWriter`, `Synthesizer`, репозитории |
| **Dependency Injection** | связывание в `main.py` (Composition Root) |
| **Information Expert** | `SpectralCube` (выборки), `ArrayGrid` (фаза наведения) |

→ Назад: [C3](C3-component.md) · [Каталог классов](../classes.md)
