# 🧩 TASK — Body-Motion · P1 (фундамент: ProjectConfig + шина DataContext + core/motion)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** новый код (config-агрегат, шина, кинематика).
> **Спека:** [`specs/body_motion_3d_2026-07-15/SPEC.md`](../specs/body_motion_3d_2026-07-15/SPEC.md) (§1,§3,§4,§7·S1).
> **Ревью спеки:** [`specs/body_motion_3d_review_2026-07-15.md`](../specs/body_motion_3d_review_2026-07-15.md) (F1,F2,F9).
> **Зависит от:** — (первый шаг P6). **Статус:** ⏳ К РЕАЛИЗАЦИИ.
>
> 🚨 🚫 pytest (только `common.runner.TestRunner`) · 🚫 писать в `.claude/worktrees/**` ·
> существующее НЕ ломать · не плодить сущности (расширяем, не переписываем).

---

> 🧭 **Конвенция 3D-визуала (везде):** дальность (range) — по **горизонтали** (пол сцены, вдаль); kx (азимут) — вбок; ky (угол места) — вверх. Динамика: GIF + `--live` окно + `.html`.

## 🎯 Цель P1

Заложить **фундамент** P6, работающий на чистом numpy (Windows): единый конфиг-агрегат,
расширенный `DataContext` с шиной (Observer), слой кинематики движения. На выходе —
траектория цели по тактам `(kx, ky, R, vr)` + 3D-визуал-подтверждение. **Без** генерации
куба (P2), помех (P3), FFT (P5), сокета (P6).

## 🧭 Опорные решения спеки (не отступать)

- **ЛЧМ и АМ — разные фронтенды** (патент гл.3/4-бис); P1 их не касается, но конфиг обязан
  описывать обе ветки (геометрия/окно/шаг — параметрами).
- **Движение без резких поворотов** (аэро-лимиты): cv + Markov-дрейф курса/скорости.
- **DataContext расширяем, шину выносим отдельно** (F1, SRP): персистентность ≠ pub/sub.
- **ProjectConfig агрегирует** существующие VO, не заменяет (F2).

## 📦 Что создать

### 1. `core/config/project_config.py` — `ProjectConfig` (Value Object, агрегат, F2)
Frozen dataclass, **агрегирует** существующие (`core/config/`): `ArrayConfig`, `RangeConfig`,
`WaveTimeConfig`, `SceneConfig` (+ новые поля P6). **Не дублировать** их поля — держать ссылки.
```python
@dataclass(frozen=True)
class ProjectConfig:
    array:  ArrayConfig          # nx, ny (расширить: non-square + pad_to_pow2, F9)
    range_: RangeConfig          # N по дальности 1024…10000
    wave:   WaveTimeConfig
    scene:  SceneConfig
    # --- P6 ---
    modulation: str = "lfm"      # "lfm" | "am"  (выбор фронтенда)
    am_window_depth: int = 16    # D (16…256), только АМ
    am_step: int = 8             # 8/16/32/64, только АМ (дефолт D/2)
    n_pulses: int = 64           # slow-time (Доплер, заглушка)
    transport_endpoint: str = "tcp://127.0.0.1:5556"
    viz_neighbor_planes: int = 5 # закладка ±N (§5)
```
- Расширить `ArrayConfig` (`core/config/array_config.py`): поддержка **non-square** (nx≠ny) и
  метод `padded_shape()` → размеры, дополненные нулями **до 2ⁿ** по каждой оси (F9). Новый
  класс **не** плодить — расширяем существующий.
- YAML для `ProjectConfig` — **A5:** через **`data_context/run_workspace.py`** (`to_yaml`/`from_yaml`/
  `config_to_dict` — свой дампер, **без PyYAML**, в офлайн-среде PyYAML нет). Расширить `config_to_dict`
  на `ProjectConfig` (агрегат) + парную сборку из dict. ⚠️ **НЕ** `YamlConfigSource` — он (а) требует
  PyYAML, (б) грузит только `WaveTimeConfig`, не `ProjectConfig`. `YamlConfigSource` оставить как есть
  (опциональный PyYAML-путь для `WaveTimeConfig`).

### 2. `core/data_context/message_bus.py` — `Blackboard` / `MessageBus` (Observer/Subject, F1)
Отдельный класс шины — **НЕ** в `DataContext` (SRP: персистентность ≠ рантайм-шина).
```python
class Observer(Protocol):
    def on_data(self, key: str, data: object) -> None: ...

class MessageBus:                     # Subject
    def subscribe(self, key: str, obs: Observer) -> None: ...
    def unsubscribe(self, key: str, obs: Observer) -> None: ...
    def publish(self, key: str, data: object) -> None: ...   # notify всех подписчиков key
```
Синхронный notify (поток-приём/queue — в P6). Ключи-каналы: `"cube"`, `"squares"`,
`"tracks"`, `"config"`.

### 3. `core/data_context/data_context.py` — **расширить** `DataContext` (F1, обратная совместимость)
- **Сохранить** `save_cube`/`load_cube` как есть.
- **Композировать** шину: `DataContext(repository=None, root=..., bus: MessageBus|None=None)`;
  свойство `.bus`. Метод-удобство `publish(key, data)` делегирует в `self._bus.publish`.
- Правило (§4): любой обмен — через `DataContext`/шину; прямого I/O в обход нет.

### 4. `core/motion/` — новый пакет кинематики
- `state.py` — `TargetState` (VO, frozen): `pos=[x,y,z]`, `vel=[vx,vy,vz]`, `acc=[ax,ay,az]`,
  `tact:int`. numpy `float64`.
- `models.py` — `MotionModel(Protocol)` + Strategy-реализации `propagate(state, dt) -> TargetState`:
  - `ConstantVelocity` — `pos += vel·dt`.
  - `MarkovDrift` — курс(az,el)/скорость как **ограниченное случайное блуждание** (OU-процесс):
    малые приращения на такт, **клип** по `max_turn_rate`, `max_accel` (аэро-лимиты). ⚠️ **без
    рывков** — параметры дрейфа малые.
  - `CoordinatedTurn` — широкий вираж (большой радиус), не рывок.
  - `ConstantAccel` — разгон/торможение вдоль вектора.
- `kinematics.py` — `Kinematics` (Pure Fabrication): `state → (az, el, R, vr)` →
  `(kx, ky, range_bin, doppler_phase)`. Геометрия апертуры (шаг `d`, λ) — из конфига.
  `vr = (r·v)/|r|` (радиальная скорость → фаза по тактам, задел под Доплер).

### 5. `core/generators/tact_sequence.py` — `TactSequence` (Iterator) — **минимальный, A3**
Класса ещё нет (в `scene.py` только `Scene`/`Synthesizer`, без итератора тактов). Создать здесь
**минимальную** версию: итератор над `MotionModel` по N тактам, `__next__` → `TargetState` +
`Kinematics`-проекция. **Реюз `Scene` как контейнер источников** (A1: НЕ плодить Composite).
Полноценное движение мульти-цели/публикация куба — расширяем в P2/P4 (не переписываем).

### 6. Демо/визуал — `demo_body_motion.py` (корень) — шаг S1
Composition Root: `ProjectConfig` (через `DataContext`) → `TactSequence` (п.5) гоняет
`MotionModel` N тактов → траектория `(kx,ky,R,vr)` → **3D-график** (plotly, реюз
`core/graphics/interactive`) в `graphics/body_motion/p1_trajectory/`. Траектория —
**реалистичная** (почти прямая + лёгкий дрейф), НЕ квадрат.

## ♻️ Реюз (точные пути)
- Конфиги/YAML: `core/config/__init__.py` (`ArrayConfig`,`RangeConfig`,`WaveTimeConfig`,
  `SceneConfig`,`YamlConfigSource`), дампер `core/data_context/run_workspace.py` (`to_yaml`/`from_yaml`).
- Фасад данных: `core/data_context/data_context.py` (расширяем), `repository.py` (не трогаем).
- 3D-визуал: `core/graphics/interactive/` (plotly), writer — `FigureWriter`/`HtmlWriter`.

## ✅ Критерии приёмки
- `ProjectConfig` агрегирует сущ. VO (не дублирует поля); `ArrayConfig` даёт non-square +
  `padded_shape()` до 2ⁿ; YAML грузится существующим лоадером.
- `MessageBus` отдельно от `DataContext`; `DataContext.save_cube/load_cube` не сломаны.
- `MotionModel`: cv + Markov-дрейф с клипом по аэро-лимитам (тест: за N тактов курс/скорость
  в пределах лимитов, траектория гладкая — без скачков >порога).
- `Kinematics`: `vr` знак верный (приближение → vr<0), `(kx,ky)` в пределах апертуры.
- Демо S1 пишет 3D-траекторию в `graphics/body_motion/p1_trajectory/`.

## 🧪 Тесты (🚫 pytest — `TestRunner`)
`tests/test_body_motion.py` (+ реестр `tests/all_test.py`): `ProjectConfigTests`,
`MessageBusTests` (publish→observer вызван), `MotionModelTests` (лимиты/гладкость),
`KinematicsTests` (vr/углы). Старые наборы — целы.

## 🚫 Границы (не делать в P1)
Куб/splat (P2), помехи (P3), мульти-цель (P4), FFT/квадраты (P5), сокет/панель (P6).
`TactSequence` в P1 — **минимальный** (траектория одной цели); публикация куба/мульти-цель — P2/P4.

## 🔧 Правки анализа тасков (2026-07-15)
A3 — `TactSequence` рождается в P1 (минимальный), не «заглушка». A5 — YAML `ProjectConfig` через
`run_workspace`, не `YamlConfigSource`. A1 — реюз `Scene`, не второй Composite.
