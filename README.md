# radar3d — модель пространственно-временно́й 3D-БПФ обработки

Каркас под предварительную сортировку сигналов на матричных ядрах GPU
(каскад 1–2 заявки). Архитектура — слои с инверсией зависимостей.

## Структура (соответствие твоим названиям)

| каталог (код)        | назначение (твоё)            |
|----------------------|------------------------------|
| `core/config/`       | `config/` — настройки модели |
| `core/data_context/` | `DataContext/` — load/save   |
| `core/generators/`   | `генератор/` — сигналы+помехи |
| `core/graphics/`     | `графика/`                   |
| `core/models/`       | `модели/` — 3FFT, далее FM…  |
| `core/controller.py` | координатор прогона          |

Имена пакетов в ASCII (PEP 8, портируемость импортов и тулинга). Кириллические
имена каталогов CPython поддерживает, но это ломает линтеры/часть инструментов —
если нужно буквально, переименование тривиально.

## Запуск

```bash
python main.py            # прогон эталонного сценария -> ./out/figures, ./out/data
```

## Паттерны

- **Strategy**: `WindowFunction`, `RadarModel`, `Visualizer`.
- **Composite**: `Scene` из `SignalSource`.
- **Abstract Factory / Registry**: `EmitterFactory` (спека → источник), OCP.
- **Builder**: `SceneBuilder`.
- **Template Method**: `RadarModel.process` (окно → преобразование → упаковка).
- **Facade**: `DataContext`.
- **Value Object**: конфиги, `SpectralCube`, `Axis`.
- **Pure Fabrication**: `FigureWriter`, репозитории.
- **Dependency Injection**: связывание в `main.py` (composition root).

## Расширение

- Новая помеха: подкласс `SignalSource` + `factory.register(Spec, builder)`.
- Новая модель (FM-корреляция, сигнал/шум): подкласс `RadarModel`, остальное не трогается.
- Новый график: подкласс `Visualizer`.
- Окно Чебышёва/Тейлора: подкласс `WindowFunction`.

## Слой классификации (каскад-классификатор)

Классификация вынесена отдельной ответственностью от спектрального
преобразования: `core/models/classification/`.

- `CubeClassifier` — абстракция (Strategy), две взаимозаменяемые реализации:
  - `RuleBasedClassifier` — детерминированный различитель по форме отклика
    (точка/гребёнка/заливка/пусто). Работает без обучения и без torch. Доказуем.
  - `Cnn3DClassifier` — обучаемая 3D-CNN (PyTorch-ROCm), адъюнкт для тяжёлых
    перекрытий и тонкого разделения (заградительная vs стороннее излучение).
- `CubeDatasetGenerator` — фабрика размеченных кубов из того же генератора сцен
  (метки идеальны, данных сколько угодно).
- `build_cnn3d()` — сеть: 2×Conv3d(stride 2) → GAP → FC, ~4 тыс. параметров.

Демо без torch:
```bash
python classify_demo.py      # по кубу на класс -> детерминированное решение
```

Обучение на машине с torch+ROCm:
```bash
python train_cnn.py --steps 400 --batch 40   # -> cnn3d.pt
```

Затем `Cnn3DClassifier("cnn3d.pt")` встаёт на место `RuleBasedClassifier` —
интерфейс тот же (LSP), остальной тракт не меняется.

Классы: empty, target, barrage, comb, ham.
