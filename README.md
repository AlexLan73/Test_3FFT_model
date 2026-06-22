# radar3d — пространственно-временна́я 3D-БПФ обработка

Каркас под предварительную сортировку сигналов на матричных ядрах GPU
(каскад 1–2 заявки). Приём квадратной антенной решёткой → **3D-БПФ**
(две угловые оси + дальность) → визуализация + классификация сцены.
Архитектура — слои с инверсией зависимостей и паттернами GoF/GRASP.

> 📚 Подробная документация — в [`Doc/`](Doc/README.md):
> [архитектура C4](Doc/architecture/README.md) · [папки](Doc/folders.md) · [классы](Doc/classes.md).

---

## Структура

| каталог | назначение |
|---------|------------|
| `core/config/` | настройки прогона (Value Objects) |
| `core/generators/` | сигналы + помехи + синтез куба |
| `core/models/` | 3D-БПФ (`Fft3DModel`) |
| `core/models/classification/` | классификатор (RuleBased / 3D-CNN) |
| `core/graphics/` | визуализаторы + запись фигур |
| `core/data_context/` | load/save кубов (Facade) |
| `core/controller.py` | координатор прогона |
| `common/` | `TestRunner` (замена pytest) |
| `tests/` | тесты под `TestRunner` |

> Имена пакетов — строчные ASCII (PEP 8): на Linux ФС регистрозависима, заглавная
> папка (`Core/`) ломает импорты.

---

## Установка

⚠️ Колёса **torch-ROCm — cp312**, поэтому venv должен быть на **Python 3.12**
(не 3.13).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .          # numpy + matplotlib + scipy
```

GPU-классификатор (опционально, из офлайн-пака):
```bash
PACK=/mnt/data/offline-debian-pack/3_python_wheels
.venv/bin/python -m pip install --no-deps \
  "$PACK/torch-rocm/torch-2.11.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl"
```
> ✅ Проверено: torch 2.11.0+rocm7.2, GPU **AMD Radeon RX 9070** (gfx1201),
> `torch.cuda.is_available()=True`. `triton-rocm` не нужен (он лишь для `torch.compile`).

---

## Запуск

```bash
python main.py            # эталонный прогон -> ./out/figures, ./out/data
python classify_demo.py   # детерминированная классификация (без torch)
python train_cnn.py --steps 400 --batch 40   # обучение -> cnn3d.pt (нужен torch)
```

Тесты (НЕ pytest — см. `.claude/rules/04-testing-python.md`):
```bash
python tests/all_test.py
```

---

## Паттерны проектирования

- **Strategy**: `WindowFunction`, `RadarModel`, `Visualizer`, `CubeClassifier`.
- **Composite**: `Scene` из `SignalSource`.
- **Abstract Factory + Registry**: `EmitterFactory` (спека → источник, OCP).
- **Builder**: `SceneBuilder`. **Template Method**: `RadarModel.process`.
- **Facade**: `DataContext`. **Value Object**: конфиги, `SpectralCube`, `Axis`.
- **Pure Fabrication**: `FigureWriter`, `Synthesizer`, репозитории.
- **Dependency Injection**: связывание в `main.py` (Composition Root).

Полный разбор связей → [Doc/architecture/C4-code.md](Doc/architecture/C4-code.md).

---

## Расширение (Open/Closed)

| Хочу добавить | Что сделать |
|---------------|-------------|
| новую помеху | подкласс `SignalSource` + `factory.register(Spec, builder)` |
| новую модель (FM-корреляция, сигнал/шум) | подкласс `RadarModel` — тракт не трогается |
| новый график | подкласс `Visualizer` |
| окно Чебышёва/Тейлора | подкласс `WindowFunction` |
| обучаемый классификатор | `Cnn3DClassifier` встаёт на место `RuleBasedClassifier` (LSP) |

---

## Классы сцены (каскад-классификатор)

`empty`, `target`, `barrage` (заградительная), `comb` (гребёнка DRFM),
`ham` (стороннее излучение / радиолюбитель).

Классификация вынесена отдельной ответственностью от спектрального преобразования
(`core/models/classification/`):

- `RuleBasedClassifier` — детерминированный различитель по форме отклика
  (точка / гребёнка / заливка / пусто). Работает без обучения и без torch.
- `Cnn3DClassifier` — обучаемая 3D-CNN (PyTorch-ROCm): `2×Conv3d(stride 2) → GAP → FC`,
  ~4 тыс. параметров. Тот же интерфейс `CubeClassifier` (LSP) — остальной тракт не меняется.
- `CubeDatasetGenerator` — фабрика размеченных кубов из того же генератора сцен
  (метки идеальны, данных сколько угодно).
