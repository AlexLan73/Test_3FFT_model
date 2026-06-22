# C2 — Container (контейнеры / пакеты)

> Из каких крупных частей состоит система. Для Python-процесса «контейнеры» —
> это логические пакеты `core.*` + точки входа + внешние библиотеки.

```mermaid
flowchart TB
    user["👤 Alex"]

    subgraph radar["radar3d (один Python-процесс)"]
        main["🚀 Точки входа<br/>main.py · classify_demo.py · train_cnn.py<br/><i>Composition Root</i>"]
        ctrl["🎛️ core.controller<br/>SimulationController<br/><i>GRASP Controller</i>"]
        config["⚙️ core.config<br/>конфиги (Value Objects)"]
        gen["🌐 core.generators<br/>сцена · источники · синтез куба"]
        models["📐 core.models<br/>3D-БПФ (Template Method)"]
        cls["🧠 core.models.classification<br/>RuleBased · 3D-CNN"]
        gfx["📊 core.graphics<br/>визуализаторы + writer"]
        data["💾 core.data_context<br/>Facade load/save"]
        common["🧪 common<br/>TestRunner"]
    end

    np["numpy / scipy"]
    mpl["matplotlib"]
    torch["torch + ROCm"]
    fs["🗂️ out/"]

    user --> main
    main --> ctrl
    main --> gfx
    main --> cls
    ctrl --> gen
    ctrl --> models
    ctrl --> data
    gen --> config
    models --> config
    cls --> models
    gen --> np
    models --> np
    gfx --> mpl
    cls --> torch
    data --> fs
    gfx --> fs

    classDef box fill:#161b22,stroke:#1f6feb,color:#fff
    classDef ext fill:#30363d,stroke:#8b949e,color:#fff
    class main,ctrl,config,gen,models,cls,gfx,data,common box
    class np,mpl,torch,fs ext
```

## Контейнеры (пакеты)

| Пакет | Ответственность | Ключевые типы |
|-------|-----------------|---------------|
| **точки входа** | связывание зависимостей (DI), CLI | `main`, `classify_demo`, `train_cnn` |
| `core.controller` | координация прогона | `SimulationController`, `ProcessingOutcome` |
| `core.config` | неизменяемые параметры прогона | `SimulationConfig`, `ArrayConfig`, `RangeConfig`, `SceneConfig`, `*Spec` |
| `core.generators` | синтез сырого куба из сцены | `Scene`, `SceneBuilder`, `Synthesizer`, `EmitterFactory`, источники |
| `core.models` | спектральное преобразование | `RadarModel`, `Fft3DModel`, `AxisWindows`, `SpectralCube` |
| `core.models.classification` | классификация куба | `CubeClassifier`, `RuleBasedClassifier`, `Cnn3DClassifier`, `CubeDatasetGenerator` |
| `core.graphics` | рендер фигур | `Visualizer`, 3 визуализатора, `FigureWriter` |
| `core.data_context` | хранение кубов | `DataContext`, `CubeRepository`, `NpyCubeRepository` |
| `common` | тест-инфраструктура (замена pytest) | `TestRunner`, `AssertionGroup`, `SkipTest` |

## Внешние зависимости

- **numpy/scipy** — массивы, БПФ, статистика (ядро тракта).
- **matplotlib** — рендер фигур (`Agg`, без GUI).
- **torch + ROCm** — только для `Cnn3DClassifier`/`train_cnn` (Python 3.12, cp312).
- **Файловая система** — `out/data` (кубы), `out/figures` (PNG), `cnn3d.pt`.

→ Назад: [C1](C1-context.md) · Дальше: [C3 — Component](C3-component.md)
