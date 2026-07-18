# 📽️ SPEC — `demo/` серия связанных примеров моделирования

> Дата: 2026-07-18 · Автор: Кодо · Статус: **черновик → ревью → Sonnet**
> Схема работы: **спека (Кодо) → глубокое ревью (Кодо) → реализация (Sonnet) → приёмка (Кодо)**.
> Скилл-оркестратор: `/demo-example` (см. `.claude/skills/demo-example/`).

---

## 0. Цель

Серия **связанных** демонстрационных примеров в новом пакете `demo/`, показывающих
сквозной тракт radar3d на нарастающих сценах. Каждый следующий пример **переиспользует**
предыдущий (сигнал / сцену / стенд). **Не изобретать** генераторы/классификаторы/арбитры —
только связывать готовые из `core/*` (см. карту в §7).

Дидактическая цепочка (ТЗ Alex):
1. **ex1** — АМ на одной временно́й оси (1D сигнал).
2. **ex2** — сигнал из ex1 размещаем в апертуре `nx×ny×N`, обработка 3FFT → куб.
3. **ex3** — данные ex2 + заградительная помеха (barrage).
4. **ex4** — то же для ЛЧМ (реюз стенда).
5. **ex5** — FM-m код: истинная цель (свежий код) vs ретранслятор (устаревший) → арбитр гл.5.

---

## 1. Именование и правила

- Пакет **строчный ASCII** — `demo/` (не `Demo/`): Linux ФС регистрозависима (правило проекта).
  Человекочитаемые названия («1. АМ на одной линии») — в `README.md`/докстринге пакета примера.
- Папки примеров без пробелов: `ex1_am_line`, `ex2_am_square`, `ex3_am_barrage`, `ex4_lfm`, `ex5_fm_m`.
- 🚫 pytest — тесты только через `common.runner.TestRunner` + `SkipTest` (правило 04).
- 🚫 писать в `.claude/worktrees/*/` (правило 03). Всё — в корень репо `demo/`.
- `demo/graphics/**` → в `.gitignore` (как `graphics/`, `out/`). PNG не коммитим.
- Стиль: ООП/SOLID/GRASP/GoF, type hints, pathlib, `np.float32/complex64`, без `print()` в
  библиотечном коде `demo/core/` (в `example.py`/`run_all.py` — можно, это Composition Root).

---

## 2. Структура каталогов

```
demo/
  __init__.py
  README.md                  # карта примеров + человекочитаемые названия + как запускать
  run_all.py                 # Composition Root: прогнать все примеры, собрать отчёты
  core/                      # ── общий стенд (Pure Fabrication) ──
    __init__.py              # реэкспорт публичного API стенда
    runner.py                # DemoRunner(ABC) — Template Method + DemoContext
    scenes.py                # SceneBank — Registry сцен-рецептов (Builder поверх core-генераторов)
    placement.py             # place_in_aperture() — steering-размещение 1D-сигнала (реюз ArrayGrid)
    writer.py                # DemoWriter — обёртка FigureWriter → demo/graphics/<example>/
    report.py                # DemoReport (Value Object) — метрики прогона
    inspect.py               # отладка: срез/дамп куба, сводка токенов/вердиктов
  ex1_am_line/
    __init__.py
    example.py               # class Ex1AmLine(DemoRunner)
    README.md
  ex2_am_square/  · example.py · README.md    # class Ex2AmSquare(DemoRunner)
  ex3_am_barrage/ · example.py · README.md    # class Ex3AmBarrage(Ex2AmSquare)  (реюз ex2)
  ex4_lfm/        · example.py · README.md    # class Ex4Lfm(DemoRunner)
  ex5_fm_m/       · example.py · README.md    # class Ex5FmM(DemoRunner)
  graphics/                  # выход PNG (gitignored), подкаталог на пример
    ex1_am_line/ …
  tests/
    __init__.py
    test_demo_core.py        # стенд: SceneBank/placement/report/writer
    test_examples.py         # приёмка каждого примера через DemoReport
    all_demo_test.py         # агрегатор (TestRunner)
```

---

## 3. `demo/core/` — общий стенд (сердце связанности)

### 3.1 `DemoRunner(ABC)` — Template Method (`runner.py`)

Единый скелет прогона. Шаги — **hook-методы**, опциональные: базовый возвращает `None`,
пример переопределяет нужное. `run()` вызывает шаги по порядку, пропуская `None`.

```python
@dataclass(frozen=True)
class DemoContext:
    """Прокидывается между шагами (SRP: состояние прогона отдельно от логики)."""
    name: str
    cfg: object                 # ProjectConfig | SimulationConfig (пример решает)
    rng: np.random.Generator

class DemoRunner(ABC):
    name: str                            # имя примера = имя подкаталога graphics
    def build_signal(self, ctx) -> SignalField | None: ...      # ex1: 1D волна
    def build_volume(self, ctx) -> np.ndarray | None: ...       # ex2+: [nx,ny,N] complex64
    def to_cube(self, ctx, volume) -> SpectralCube | None: ...  # 3FFT / WaveformToCube
    def tokenize(self, ctx, cube) -> tuple[list, list] | None:  # (tokens, verdicts)
    def arbitrate(self, ctx, verdicts) -> list | None: ...      # list[TargetDecision]
    def classify(self, ctx, cube) -> Classification | None: ... # RuleBasedClassifier
    def visualize(self, ctx, *, signal, cube, ...) -> dict[str, Figure]: ...  # {png_name: Figure}
    def run(self) -> DemoReport:         # оркестратор (final — не переопределять)
        ...
```

- `run()` строит `DemoContext`, гонит шаги, пишет фигуры через `DemoWriter(self.name)`,
  собирает `DemoReport`. **Детерминизм**: `rng = np.random.default_rng(seed)` (seed в cfg/поле).
- Template Method: подкласс переопределяет **только** свои шаги. ex1 — лишь `build_signal`+`visualize`.

### 3.2 `SceneBank` — Registry сцен-рецептов (`scenes.py`)

Набор именованных **тестируемых сцен**. Рецепт = callable, возвращающий сигнал/volume,
**реюзящий** core-генераторы. Обеспечивает связь ex1→ex2→ex3.

```python
class SceneBank:
    _recipes: dict[str, Callable[[SceneParams], SceneResult]]
    @classmethod
    def register(cls, name, recipe) -> None: ...
    @classmethod
    def get(cls, name) -> Callable: ...
    @classmethod
    def names(cls) -> list[str]: ...
```

Сцены (реюз §7):
| имя | что | реюз |
|-----|-----|------|
| `am_line`    | 1D АМ сигнал | `WaveformFactory().create(Modulation.AM).render(...)` |
| `am_square`  | АМ-цель в апертуре nx×ny×N | `am_line` + `place_in_aperture(...)` |
| `am_barrage` | `am_square` + заград с угла | `am_square` + `BarrageRfJammer().render(...)` (суммирование в volume) |
| `lfm_target` | ЛЧМ-цель в апертуре | `LfmWaveform` + `place_in_aperture` (или `build_lfm_target_volume`) |
| `fm_m_dual`  | цель (свежий код) + ретранслятор (устаревший) | `m_sequence_pow2` + `PhaseCodeWaveform` + два угла |

`am_barrage` **вызывает** `am_square` внутри → связь примеров на уровне данных, не копипаст.

### 3.3 `place_in_aperture()` — размещение (`placement.py`)

Тонкая обёртка, **не** новый генератор. Размножает 1D-сигнал по решётке через steering:

```python
def place_in_aperture(signal_1d: np.ndarray, kx: float, ky: float,
                      array: ArrayConfig) -> np.ndarray:
    """signal[N] → volume[nx,ny,N]: steering-фаза цели (kx,ky) на решётке.
    Реюз ArrayGrid.from_config(array).steering(kx,ky) — веса [nx,ny]."""
```
Возврат `complex64`, не мутирует вход.

### 3.4 `DemoWriter` (`writer.py`)

Обёртка над `FigureWriter`: путь `demo/graphics/<example>/`, сам создаёт каталог (Pure Fabrication).
```python
class DemoWriter:
    def __init__(self, example: str, root: Path = Path("demo/graphics")): ...
    def write(self, fig, name: str) -> str: ...   # делегирует FigureWriter.write
```

### 3.5 `DemoReport` — Value Object (`report.py`)

```python
@dataclass(frozen=True)
class DemoReport:
    example: str
    figures: list[str]                      # пути PNG
    n_tokens: int = 0
    verdicts: tuple = ()                    # RangeVerdict kind'ы
    decisions: tuple = ()                   # TargetDecision
    classification: Classification | None = None
    metrics: dict = field(default_factory=dict)   # доп. числа (SNR, пики...)
    def __str__(self) -> str: ...           # человекочитаемая сводка для консоли/теста
```
Используется тестами-приёмкой (§5) — детерминированные проверки без глаз.

### 3.6 `inspect.py` — отладка (опц.)

`dump_cube_slice(cube, r) -> str`, `summarize_tokens(tokens, verdicts) -> str`. Только чтение.

---

## 4. Примеры (что делает каждый, какой шаг переопределяет)

### ex1 — `Ex1AmLine(DemoRunner)`  ·  «АМ на одной временно́й оси»
- Переопределяет: `build_signal` (сцена `am_line`), `visualize`.
- Сигнал: `WaveformFactory().create(Modulation.AM).render(NumpyBackend(), WaveformSpec(fs, carrier_hz, n_samples, meta={"m":…,"f_m":…}), rng)`; берём `field.data[0,0,:]`.
- Графики: `signal_time.png` (Re/огибающая), `signal_spectrum.png` (|FFT|, дБ).
- Report: `metrics={"n_samples", "carrier_hz", "m"}`.

### ex2 — `Ex2AmSquare(DemoRunner)`  ·  «сигнал в апертуре nx×ny×N»
- Переопределяет: `build_volume` (сцена `am_square` = ex1-сигнал через `place_in_aperture`), `to_cube`, `classify`, `visualize`.
- Дефолт `array=ArrayConfig(64,64)`, `N=4096`; CLI `--nx/--ny/--n` до 512×256×10000.
- Куб: `AmToCube(...).fill(volume, cfg)` **или** `Fft3DModel(...).process(volume)` — выбрать по факту (АМ = локальный 3D-FFT, §P5). Sonnet обосновывает выбор в докстринге.
- Графики: `cube_scatter.png` (`CubeScatterVisualizer`), `angular_map.png`.
- Report: `classification` (ожидаем `target`), `metrics={"nx","ny","N","peak_kx","peak_ky"}`.

### ex3 — `Ex3AmBarrage(Ex2AmSquare)`  ·  «+ заградительная помеха»
- **Наследует ex2**, переопределяет `build_volume` (сцена `am_barrage` = `am_square` + `BarrageRfJammer`), `tokenize`, `arbitrate`.
- Токенизатор: `VolumeTokenizer(window_l=1).tokenize(cube)` → `assemble_range(...)` → `EdgeArbiter().arbitrate(...)`.
- Ожидание (приёмка): среди вердиктов есть `barrage`; цель на своём угле выживает (`target`/`jammer` разделены арбитром).
- Графики: `cube_scatter.png`, `angular_map.png`, `tokens.png` (вердикты по дальности).

### ex4 — `Ex4Lfm(DemoRunner)`  ·  «ЛЧМ через тот же стенд»
- Сцена `lfm_target`; куб через `LfmToCube(...).fill(volume, cfg)` (дечирп + RangeFft + angular_fft).
- Показывает, что стенд не завязан на АМ (OCP): смена модуляции = смена сцены, не тракта.
- Графики: `cube_scatter.png`, `range_profile.png` (`RangeProfileVisualizer`).

### ex5 — `Ex5FmM(DemoRunner)`  ·  «FM-m код: цель vs ретранслятор»
- Сцена `fm_m_dual`: истинная цель (kx,ky) с **текущим** кодом `m_sequence_pow2(deg, seed=now)`;
  ретранслятор на др. угле с **устаревшим** кодом. Прототип — `Example/Hener1/files 9/code_bank.py`.
- Арбитр: `CombinedArbiter(EdgeArbiter(), CodeArbiter(ref_code=<current>, signal_by_angle={...}))`.
- Ожидание (приёмка): цель → `decision="target"`, ретранслятор → `"false"`/`"jammer"`.
- Графики: `code_bank.png` (корреляция по углам), `decisions.png`.

---

## 5. Тесты (TestRunner, 🚫 pytest)

- `tests/test_demo_core.py` — `SceneBank.register/get`, `place_in_aperture` (форма/steering-фаза),
  `DemoReport.__str__`, `DemoWriter` пишет в правильный подкаталог (temp).
- `tests/test_examples.py` — прогон каждого примера на **лёгком** дефолте, проверка `DemoReport`:
  - ex1: `n_samples>0`, спектр непустой.
  - ex2: `classification.name == "target"`.
  - ex3: `"barrage" in verdicts`.
  - ex5: цель `target`, ретранслятор `false/jammer`.
  - GPU/torch-шаги — под `SkipTest`, если нет.
- `tests/all_demo_test.py` — агрегатор: `for cls in [...]: cls().run_all()`.
- Размеры в тестах — минимальные (напр. 16×16×256), чтобы бегало дома без scipy/GPU.

---

## 6. `run_all.py` — Composition Root

```python
examples = [Ex1AmLine(), Ex2AmSquare(), Ex3AmBarrage(), Ex4Lfm(), Ex5FmM()]
for ex in examples:
    report = ex.run()          # пишет PNG в demo/graphics/<ex>/
    print(report)              # сводка
```
CLI (argparse): `--only ex2`, `--nx/--ny/--n` (масштаб ex2), `--seed`.

---

## 7. Карта реюзаемых компонентов (НЕ изобретать заново)

| Слой | Класс/функция | Файл |
|------|---------------|------|
| Волны | `WaveformFactory.create(Modulation.*)`, `.render(backend, WaveformSpec, rng)` | `core/generators/waveforms/factory.py` |
| Модуляции | `Modulation.{CW,LFM,AM,PHASE_CODE,FM_INTERFERENCE,BARRAGE,SMSP,DRFM_REPEATER,...}` | `.../waveforms/base.py` |
| Помеха barrage | `BarrageRfJammer().render(...)` | `.../waveforms/jammers_rf.py` |
| ФМн/M-код | `PhaseCodeWaveform`, `m_sequence_pow2(degree, seed)` | `.../waveforms/phase_code.py`, `mseq.py` |
| Бэкенд | `NumpyBackend()`, `HipBackend()` (опц. GPU) | `core/generators/backends/` |
| Решётка | `ArrayGrid.from_config(array).steering(kx,ky)` | `core/generators/grid.py` |
| Сцена (куб-ветка) | `SceneBuilder().build(SceneConfig)`, `Synthesizer(array,rng,seed).build(scene)` | `core/generators/{factory,scene}.py` |
| 3FFT | `Fft3DModel(array,rng,windows).process(volume) -> SpectralCube` | `core/models/fft3d.py` |
| Волна→куб | `LfmToCube(...).fill(volume,cfg)`, `AmToCube(...).fill/scan` | `.../waveforms/waveform_to_cube.py` |
| Куб VO | `SpectralCube.{magnitude,magnitude_db,index_of_angle,range_profile_db,kx,ky,range}` | `core/models/result.py` |
| Токенизатор | `VolumeTokenizer(window_l=1).tokenize(cube)`, `assemble_range(tokens)` | `core/models/tokenizer/tokenizer.py` |
| Арбитр гл.5 | `EdgeArbiter()`, `CodeArbiter(ref_code,signal_by_angle)`, `CombinedArbiter(edge,code)`, `fm_correlate` | `core/models/tokenizer/arbiter.py` |
| Классификатор | `RuleBasedClassifier().classify(cube) -> Classification` | `core/models/classification/rule_based.py` |
| Графика | `CubeScatterVisualizer`, `AngularMapVisualizer`, `RangeProfileVisualizer`, `FigureWriter` | `core/graphics/` |
| Config | `ArrayConfig(nx,ny)`, `RangeConfig(n_real,n_fft)`, `ProjectConfig`, `default_scenario()` | `core/config/` |

Прототипы на «салфетке» (эталоны для сверки, read-only): `Example/Hener1/files 8/scenario.py`
(≙ сцена), `files 9/code_bank.py` (≙ FM-m арбитр).

---

## 8. GoF/SOLID карта примеров

| Паттерн | Где |
|---------|-----|
| Template Method | `DemoRunner.run()` (скелет), подклассы — шаги |
| Registry/Factory | `SceneBank` (сцены по имени) |
| Builder | сцены-рецепты (сборка сигнала/volume) |
| Pure Fabrication | `DemoWriter`, `placement`, `inspect` (нет доменной сущности) |
| Value Object | `DemoContext`, `DemoReport` (frozen) |
| Strategy | реюз `Visualizer`/`WaveformToCube`/`Arbiter` (готовые) |
| Composite | `CombinedArbiter` (ex5), `Scene` (при куб-ветке) |
| DI | связывание в `run_all.py`/`example.py` (Composition Root), без глобалов |
| LSP | `Ex3AmBarrage(Ex2AmSquare)` — подстановка без слома; `EdgeArbiter`↔`CombinedArbiter` |
| OCP | новая модуляция = новая сцена, тракт не трогаем (ex4 доказывает) |

---

## 9. Приёмка (Кодо проверяет после Sonnet)

1. Все каталоги/файлы по §2, пакет строчный, нет пробелов в именах.
2. `python demo/run_all.py` на дефолте (дома) — зелёный, PNG в `demo/graphics/<ex>/`.
3. `python demo/tests/all_demo_test.py` — 0 fail (GPU/torch под SkipTest).
4. Реюз: **нет** новых генераторов/классификаторов/арбитров — только связывание §7 (grep на дубли).
5. ex3 детектит `barrage`; ex5 различает цель/ретранслятор (по DemoReport).
6. `demo/graphics/` в `.gitignore`. Нет записи в `.claude/worktrees/`.
7. ruff (если есть) чистый; type hints; нет `print()` в `demo/core/`.

---

*Связка: TASK_demo_examples_p1.md · IN_PROGRESS.md · скилл `/demo-example`.*
