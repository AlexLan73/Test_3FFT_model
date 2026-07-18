# radar3d — пространственно-временна́я 3D-БПФ обработка + когнитивный тракт

Предварительная сортировка сигналов на матричных ядрах GPU (каскад 1–2 заявки).
Приём антенной решёткой **i×j** (нечётная апертура, каждая ось 2ⁿ) → **два FFT**
(дальностный + угловой) / скользящий 3D-FFT → куб «угол×угол×дальность» → **токенизатор**
(признаки + триаж) → **арбитр** (передний край τ≥0 + свежесть FM-m кода) → **целеуказание**
пучка → **трекинг** между тактами. Архитектура — слои с инверсией зависимостей, GoF/GRASP.

> 📚 Подробная документация — в [`Doc/`](Doc/README.md):
> [архитектура C4](Doc/architecture/README.md) · [папки](Doc/folders.md) · [классы](Doc/classes.md).
> Патент/статья — [`Doc/Patent/`](Doc/Patent/) (главы 0–9 + формула + заявки).

---

## Апертура i×j / 2ⁿ / zero-pad

Апертура **не обязана быть квадратной**: `nx≠ny` допустимо, каждая ось дополняется нулями
до ближайшей степени двойки (`ArrayConfig.padded_shape()`). Угловой FFT паддит апертуру до 2ⁿ
перед `fft2`; шкала `sinθ = k/(N_pad/2)` по каждой оси независимо (частный случай 16×16 → k/8).
Признаки токенизатора нормированы на число ячеек M = N_pad_x·N_pad_y (инвариантность к апертуре).
Детали — [`Doc/Patent/00_КОНЦЕПЦИЯ_ixj_2n.md`](Doc/Patent/00_КОНЦЕПЦИЯ_ixj_2n.md).

---

## Конвейер (когнитивный такт)

```
сигнал ─► фронтенд (Strategy):
            ЛЧМ → 2 раздельных FFT (дальностный глобальный + угловой i×j поячеечно)   [точно]
            АМ  → скользящий 3D-FFT по окну nx×ny×D                                     [грубо]
       ─► куб «угол×угол×дальность» N_pad_x×N_pad_y×L      (общий примитив)
       ─► ТОКЕНИЗАТОР: OS-CFAR (точная Pfa, Rohling) → 6 признаков (норм. на M)
            → триаж {noise/source/smeared} → проход 2 (target/comb/barrage по дальности)
       ─► АРБИТР гл.5: передний край τ≥0 (геометрия) + свежесть FM-m кода (CodeArbiter)
            → CombinedArbiter (чистит осколки заграда)
       ─► ЦЕЛЕУКАЗАНИЕ гл.8: пучок лучей FM-m в конус неопределённости
       ─► ТРЕКИНГ между тактами (§4-бис.4 «летит»): NearestNeighborTracker
anti-barrage: SubspaceNuller (проекция) / RobustMvdrNuller (Capon) → Pipeline → ROI-гейт → кластеризация
```

---

## Структура

| каталог | назначение |
|---------|------------|
| `core/config/` | `ProjectConfig`/`ArrayConfig`(i×j)/`RangeConfig`/… (Value Objects) |
| `core/generators/` | сигналы + помехи + синтез куба; `waveforms/` (CW/ЛЧМ/АМ/ФМн/ЧМ + jammers + mseq) |
| `core/models/` | `Fft3DModel`, `angular_fft`/`range_fft` (2 FFT), `waveforms/waveform_to_cube` |
| `core/models/tokenizer/` | `VolumeTokenizer`, `FeatureExtractor`, `OsCfarDetector`, триаж, **`arbiter`** (гл.5) |
| `core/models/targeting/` | `BeamTargeting`/`CognitiveCycle` (гл.8), `RoiGate` |
| `core/models/tracking/` | `NearestNeighborTracker`, `Track` (движение из треков) |
| `core/models/anti_barrage/` | `SubspaceNuller`, `RobustMvdrNuller`, `AntiBarragePipeline`, CFAR, кластеризация |
| `core/models/classification/` | `RuleBasedClassifier` / `Cnn3DClassifier` (torch, опц.) |
| `core/graphics/` | визуализаторы + `panel/` (Dear PyGui, опц.) |
| `core/data_context/` | load/save кубов (Facade) + `MessageBus` (Observer) |
| `core/runtime/` | `SceneServer`, `Transport` (ZMQ), `codec` (msgpack) |
| `core/motion/` | `MotionModel`/`Kinematics`/`TargetState` (движение цели) |
| `core/snr/` | SNR-эстиматоры (спектр CA-CFAR + статистика) |
| `common/` | `TestRunner` (замена pytest) · `tests/` — 39 наборов |

> Имена пакетов — строчные ASCII: на Linux ФС регистрозависима (`Core/` ломает импорты).

---

## Установка

⚠️ Колёса **torch-ROCm — cp312**, venv на **Python 3.12** (не 3.13).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .          # numpy + matplotlib (+ scipy опц.)
```
Весь numpy-бэкенд работает **без scipy/torch/GPU** (Windows дома). GPU-классификатор — опц.:
```bash
PACK=/mnt/data/offline-debian-pack/3_python_wheels
.venv/bin/python -m pip install --no-deps \
  "$PACK/torch-rocm/torch-2.11.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl"
```
> ✅ torch 2.11.0+rocm7.2, AMD Radeon RX 9070 (gfx1201). `triton-rocm` не нужен.

---

## Запуск

```bash
python main.py            # эталонный прогон -> ./out/figures, ./out/data
python classify_demo.py   # детерминированная классификация (без torch)
python demo_tokenizer.py --nx 6 --ny 15   # токенизатор на неквадратной апертуре
python train_cnn.py --steps 400 --batch 40   # обучение (нужен torch+GPU)
```
Тесты (🚫 pytest — см. `.claude/rules/04-testing-python.md`): `python tests/all_test.py` (39 наборов).

---

## Паттерны проектирования

- **Strategy**: `WindowFunction`, `RadarModel`, `Visualizer`, `Arbiter`, `Targeting`, `Tracker`.
- **Template Method**: `RadarModel.process`, `VolumeTokenizer.tokenize`. **Facade**: `DataContext`,
  `AntiBarragePipeline`, `CognitiveCycle`. **Composite**: `Scene`, `CombinedArbiter`, `FanOutTransport`.
- **Abstract Factory/Registry**: `WaveformFactory`, `EmitterFactory`. **Builder**: `SceneBuilder`.
- **Value Object**: конфиги, `SpectralCube`, `SliceToken`, `TargetDecision`, `BeamCommand`, `Track`.
- **Observer**: `MessageBus`, `SceneServer`/`Transport`. **DI**: связывание в Composition Root.

Полный разбор → [Doc/architecture/C4-code.md](Doc/architecture/C4-code.md).

---

## Расширение (Open/Closed)

| Хочу добавить | Что сделать |
|---------------|-------------|
| новый тип сигнала (фронтенд) | подкласс `WaveformToCube` — куб-примитив и токенизатор не трогаются |
| новую помеху | подкласс `Waveform`/`SignalSource` + `factory.register(...)` |
| обучаемый триаж | MLP на место `RuleBasedTriage` (тот же `SliceTriage`, LSP) |
| код-арбитр по FM-m | `CodeArbiter` (реализован) / свой `Arbiter` |
| робастный nuller | `RobustMvdrNuller` (Capon + diagonal loading) на место `SubspaceNuller` |
| обучаемый классификатор | `Cnn3DClassifier` на место `RuleBasedClassifier` (LSP) |

---

## Классы сцены

`empty`, `target`, `barrage` (заградительная), `comb` (гребёнка DRFM), `ham` (стороннее / RFI).
Проход 1 токенизатора различает {noise/source/smeared}; проход 2 — {target/comb/barrage} по дальности;
арбитр гл.5 ставит финальную метку цель/ложь (передний край + свежесть кода).
