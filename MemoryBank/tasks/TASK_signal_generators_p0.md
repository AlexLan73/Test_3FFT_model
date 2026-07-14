# 🧩 TASK — Генераторы сигналов · P0 (фундамент: SignalField + конфиг + окно + YAML)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** новый код (не порт), ООП/SOLID/GoF.
> **Спека:** [`specs/signal_generators_2026-07-13.md`](../specs/signal_generators_2026-07-13.md) (§0,§4,§5.1,§9).
> **Ревью-риски:** [`specs/signal_generators_review_2026-07-14.md`](../specs/signal_generators_review_2026-07-14.md).
> **Статус:** ✅ РЕАЛИЗОВАНО (Sonnet) + ✅ ПРИНЯТО (ревью Кодо 2026-07-14). Тесты 11 ok + 1 skip (pyyaml
> вне .venv; YAML-путь сверен на cp313), старые 28 целы, ruff/mypy чисто, визуал ок, изменения аддитивные.
> Хвост для P1: убрать заглушку `GenBackend` из `waveforms/base.py` → настоящий в `backends/base.py`.
>
> 🚨 **КРИТИЧНО:** 🚫 `pytest` (только `common.runner.TestRunner` + `AssertionGroup` + `SkipTest`).
> 🚫 писать в `.claude/worktrees/**`. Всё — в корень репо. Существующий `core/generators/*` (куб-уровень)
> и `core/config/*` **НЕ ломать** — только добавлять.

---

## 🎯 Цель P0

Заложить **фундамент данных** слоя генерации: абстрактный носитель `SignalField`, VO-спеки волны и окна,
расширенный конфиг сырого времени и **YAML-ввод**. **Без GPU и без формул сигналов** (это P1/P2) —
только структуры данных + загрузка конфигов. Всё работает на чистом numpy (Windows-совместимо).

## 📦 Что создать (файлы + сигнатуры)

Новый подпакет `core/generators/waveforms/` (см. §9 спеки). Все VO — `@dataclass(frozen=True)`,
type hints, `from __future__ import annotations`, pathlib.

### 1. `core/generators/waveforms/field.py` — `SignalField` (VO, абстрактный носитель, §4.0)
```python
class Modulation(Enum): AM, LFM, PHASE_CODE, FM_INTERFERENCE, ...   # тип модуляции
class AxisKind(Enum): ANTENNA_X, ANTENNA_Y, FAST_TIME, RANGE_BIN, CORR_DELAY, ...  # смысл оси

@dataclass(frozen=True, eq=False)     # ⚠️ eq=False — см. подводный камень G1 ниже
class SignalField:
    data: np.ndarray                 # payload, сейчас сырое время [nx, ny, n_samples] complex64
    modulation: Modulation
    axes: tuple[AxisKind, ...]       # смысл каждой оси data (len == data.ndim)
    fs: float                        # Гц
    carrier_hz: float                # несущая/IF
    tact: int = 0                    # индекс такта (§4.5)
    meta: Mapping[str, float] = field(default_factory=dict)  # ΔF, snr_db, m, f_m, kx, ky …
    def __post_init__(self):         # валидация: len(axes)==data.ndim; dtype complex64
        if len(self.axes) != self.data.ndim: raise ValueError(...)
        # frozen → присваивание только через object.__setattr__ (напр. привести dtype/заморозить meta):
        # object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
```
- **Смысл (§4.0, ответ Q6):** данные у всех типов формируются одинаково (сырое время), различие —
  в `modulation`/`axes`/`meta`. Не зашивать «третья ось = дальность» (это только ЛЧМ).
- `data` **не мутировать** (чистота, R6). complex64 (экономия под GPU).
- 🐞 **G1 (баг, ревью тасков):** `@dataclass(frozen=True)` по умолчанию генерит `__eq__`, который сравнит
  `data` как массивы → любой `field == other` бросит *«truth value of an array is ambiguous»*, а `__hash__`
  упадёт на unhashable ndarray. → `eq=False` (identity-семантика VO-носителя) **или** `data: … = field(compare=False)`.
- 🐞 **G2:** `meta` — `field(default_factory=dict)` (не мутабельный дефолт); для неизменяемости VO
  оборачивать в `MappingProxyType` в `__post_init__` (через `object.__setattr__`, т.к. frozen).

### 2. `core/generators/waveforms/base.py` — `Waveform(ABC)` + `WaveformSpec` (VO)
```python
@dataclass(frozen=True)
class WaveformSpec:
    fs: float; carrier_hz: float; n_samples: int
    amplitude: float = 1.0; phase: float = 0.0
    fdev_hz: float = 0.0             # девиация/полоса ЛЧМ
    snr_db: float | None = None      # R5: калибровка по SNR (None = амплитуда как есть)
    tau_s: float = 0.0               # задержка
    window: "TimeWindow" = ...       # размещение (см. ниже)
    meta: Mapping[str, float] = field(default_factory=dict)  # G10: параметры под тип — AM: m,f_m; …

class Waveform(ABC):                 # Strategy
    @abstractmethod
    def render(self, backend: "GenBackend", spec: WaveformSpec,
               rng: np.random.Generator) -> SignalField: ...
```
- **P0 только объявляет ABC и WaveformSpec** — конкретные `AmWaveform/LfmWaveform/...` в P1.
- `render` принимает `rng` явно (R6: детерминизм).

### 3. `core/generators/waveforms/placement.py` — `TimeWindow` (VO + Decorator, §4.4)
```python
@dataclass(frozen=True)
class TimeWindow:
    kind: Literal["full", "partial", "short"]
    t0: float = 0.0; t1: float | None = None; dur: float | None = None
    def mask(self, n_samples: int, fs: float) -> np.ndarray:   # bool[n_samples]
        ...  # full → все True; partial → [t0,t1]; short → [t0, t0+dur]
```
- Аналог `in_window` из `getX_numpy` (DSP-GPU `factories.py:87`). Энергия **вне маски = 0**.

### 4. `core/config/` — расширение (НЕ ломать существующее)
- `core/config/waveform_config.py` — `WaveTimeConfig` (VO): дефолты из **§5.1 спеки** (baseline):
  `fs=12e6, carrier_hz=2e6, fdev_hz=6e6, n_samples=8192, nx=16, ny=16, seed=7`.
  Реюз существующего `ArrayConfig` (nx,ny) — не дублировать геометрию.
- `core/config/config_source.py` — `ConfigSource` (Strategy/Facade): грузит `configs/*.yaml`
  **в существующие/новые VO** (`WaveTimeConfig`, `*Spec`). НЕ второй конфиг-слой (R10).
  ```python
  class ConfigSource(Protocol):
      def load(self) -> WaveTimeConfig: ...
  class YamlConfigSource(ConfigSource): ...   # pyyaml → VO
  class DefaultConfigSource(ConfigSource): ...  # baseline из §5.1 (для Windows/без yaml)
  ```
- `core/config/configs/baseline.yaml` — стартовый набор чисел из §5.1 (Alex будет править).
- **Sweep-хук (Q2):** `ConfigSource` умеет отдавать **последовательность** конфигов для перебора
  (генератор датасета) — метод `iter_configs() -> Iterator[WaveTimeConfig]` (по умолчанию — один).

## ♻️ Реюз (точные пути — читать перед кодом)
- Окно-маска: `DSP-GPU/DSP/Python/signal_generators/factories.py:87` (`getX_numpy.in_window`).
- Геометрия n×n: наш `core/generators/grid.py` (`ArrayGrid`, `steering`) — реюз, не дублировать.
- Спеки-VO паттерн: наш `core/config/scene_config.py` (`@dataclass(frozen=True)`).

## 🖼️ Визуал-подтверждение (обязательно, §9-конвенция)
Каталог: **`graphics/signal_generators/p0_field_window/`** (через `FigureWriter`, сам создаёт путь).
- `window_masks.png` — 3 сабплота масок `TimeWindow`: `full` / `partial(t0,t1)` / `short(t0,dur)`
  поверх шкалы времени (наглядно: где сигнал есть, где 0). Это доказывает, что размещение (§0.3) работает
  ещё до формул сигналов. Демо-точка — в `demo_generators.py` (или временный `demo_p0.py`, удалить после P1).

## ✅ Definition of Done
- `python -c "import core.generators.waveforms"` — импортируется на **cp313** (system numpy 2.2, R9)
  **и cp312** (ядро version-agnostic, §4.3 — чистый numpy, без GPU/torch).
- `tests/test_generators.py` (новый, через `common.runner.TestRunner`):
  - `SignalField.__post_init__` валидирует `len(axes)==ndim`, dtype complex64 (raise на нарушении).
  - `SignalField.__eq__`/хэш **не падают** (G1): сравнение/использование как ключа не бросает по ndarray.
  - `TimeWindow.mask`: full→все True; partial(t0,t1)→корректный срез; short→длина `round(dur·fs)`;
    энергия вне маски = 0.
  - `YamlConfigSource` грузит `baseline.yaml` → `WaveTimeConfig` с числами §5.1.
  - `DefaultConfigSource` даёт те же дефолты без pyyaml (Windows-путь).
- 🖼️ `graphics/signal_generators/p0_field_window/window_masks.png` создан.
- Реестр — добавить набор в `tests/all_test.py`.
- `ruff check core/generators core/config` — чисто. `mypy core/generators/waveforms` — 0 ошибок.

## ⚠️ Подводные камни (из ревью)
- **R6:** любые rng — через **переданный** `np.random.Generator`, никаких глобальных `np.random.*`.
- **R9:** numpy 2.2 — не использовать `np.complex_`/`np.float_` (удалены); явный `np.complex64`.
- **R10:** `pyyaml` может отсутствовать → `YamlConfigSource` при `ImportError` даёт понятную ошибку,
  а `DefaultConfigSource` работает без него (тест YAML — под `SkipTest`, если нет pyyaml).
- **НЕ трогать** `core/generators/sources.py|jammers.py|scene.py` (куб-уровень) и `core/config/scene_config.py`
  сверх добавления новых спек.

## 🚫 Вне P0
GPU-бэкенд, формулы сигналов (CW/ЛЧМ/АМ/ФМн/ЧМ), помехи, дечирп, коррелятор, движение/такты — это P1+.
