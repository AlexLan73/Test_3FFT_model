# Ревью: core/motion + core/snr (ООП/SOLID/GRASP/GoF)

Кластер: `core/motion/{state,models,kinematics}.py`, `core/snr/{config,estimator,signal}.py`
(+ `__init__.py` реэкспорты). Только чтение, код не менялся.

## Находки (severity ↓)

### [HIGH] VO immutability / aliasing — `core/motion/state.py:9-12`

**Что:**
```python
def _vec3(value: np.ndarray | tuple[float, float, float] | list[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    arr.setflags(write=False)   # VO: состояние неизменяемо после конструирования
    return arr
```
`np.asarray(value, dtype=np.float64)` **не копирует**, если `value` уже
`np.float64`-массив нужной формы — возвращает представление (`view`), которое
делит буфер памяти с исходным массивом вызывающего кода. `.reshape(3)` тоже
возвращает view, а не copy. `setflags(write=False)` запрещает запись только
через *этот* view/объект `TargetState.pos` — но не через оригинальный массив,
который остался у вызывающего и полностью writable.

Подтверждено эмпирически:
```python
a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
st = TargetState(pos=a)
a[0] = 999.0
st.pos  # -> [999. 2. 3.]   <-- "неизменяемое" состояние поменялось извне!
np.shares_memory(st.pos, a)  # True
```

**Почему нарушение:** docstring класса прямо заявляет «состояние неизменяемо
после конструирования» (VO-инвариант, `@dataclass(frozen=True)`). Через
aliasing это неверно для float64-входа нужной формы — тихая порча состояния
задним числом, без исключения, без предупреждения. Также нарушает правило
проекта «не мутировать входные массивы» (`05-python-style.md`) в обратную
сторону: VO не защищается от чужой последующей мутации, потому что не делает
собственную копию при построении.

**Как исправить:** форсировать копию в `_vec3`, независимо от dtype/shape:
```python
arr = np.array(value, dtype=np.float64, copy=True).reshape(3)
```
(`np.array(..., copy=True)` вместо `np.asarray`). Стоит добавить unit-тест
именно на этот сценарий (мутация исходного массива после конструирования
`TargetState` не должна быть видна в `state.pos`), т.к. дефект тихий.

---

### [MEDIUM] LSP — `SnrEstimator` Protocol / `StatisticsSnrEstimator` — `core/snr/estimator.py:54-67, 245-269`

**Что:** `SnrEstimator.estimate(signal, support: slice | None = None)` объявляет
`support` опциональным (default `None`) — по контракту базового протокола
вызов `estimator.estimate(signal)` валиден для любой реализации.
`SpectrumSnrEstimator.estimate` действительно игнорирует `support`.
Но `StatisticsSnrEstimator.estimate` **усиливает предусловие**: `if support is
None: raise ValueError(...)` (estimator.py:265-269) — то есть тот же вызов
`estimator.estimate(signal)`, который работает для одной реализации Protocol,
падает для другой.

**Почему нарушение:** классический LSP-смёлл — подтип не должен ужесточать
предусловие базового контракта. Клиентский код, написанный против абстракции
`SnrEstimator` (например, обобщённый цикл `for est in estimators:
est.estimate(sig)`), не может считать обе реализации взаимозаменяемыми без
доп. знания о конкретном классе. Сейчас это спасает то, что все текущие
вызывающие места (`demo_snr.py`, `tests/test_snr.py`) знают конкретный тип и
осознанно передают/не передают `support` — но сам протокол лжёт о своём
контракте.

**Как исправить:** один из вариантов: (а) сделать `support` обязательным
позиционным параметром в самом Protocol (раз он не опционален по факту для
одной ветки) и обновить `SpectrumSnrEstimator`, чтобы он просто игнорировал
непустое значение явно; либо (б) вынести два разных протокола
(`SnrEstimator` без support / `SupportRequiredSnrEstimator` с обязательным
support) — ISP вместо одного протокола с молчаливо разной семантикой.
Текущее поведение уже задокументировано в docstring (estimator.py:56-59) —
это смягчает, но не отменяет находку.

---

### [MEDIUM] DRY / дублирование пайплайна — `core/snr/estimator.py:121-161` vs `194-215`

**Что:** `SpectrumSnrEstimator.estimate()` (decimation → window → zero-pad →
FFT → `|X|²`, шаги 1-4, строки ~143-161) и `SpectrumSnrEstimator.get_mag_sq()`
(строки ~204-214) — **буквально одна и та же последовательность операций**,
скопированная в два метода одного класса.

**Почему нарушение:** источник истины на шаги пайплайна раздвоен внутри
одного класса. Именно этот класс дефектов — два места с одной и той же
формулой — уже описан как прецедент проекта в `.claude/rules/07-math-in-core.md`
(3D-проекция в demo, «знаки заданы в ДВУХ местах → рассинхрон»). Здесь риск
тот же на уровне класса: правка шага пайплайна (например, добавление нового
окна или изменение zero-pad) в `estimate()` без синхронной правки в
`get_mag_sq()` — и график (`get_mag_sq`, используется для диагностики/
отрисовки спектра) начнёт расходиться с фактическим SNR-расчётом (`estimate`).

**Как исправить:** вынести шаги 1-4 в приватный метод
`_prepare_spectrum(self, signal) -> tuple[np.ndarray, int, int]`
(mag_sq, n_actual, n_fft), вызываемый из `estimate()` и `get_mag_sq()`.

---

### [MEDIUM] DRY — интеграция позиции/ускорения дублируется в 3 стратегиях — `core/motion/models.py`

**Что:** одна и та же пара строк (полу-неявное интегрирование по времени)
повторена дословно:

- `MarkovDrift.propagate` — models.py:87-88
- `CoordinatedTurn.propagate` — models.py:109-110
- `WeavingManeuver.propagate` — models.py:171-172

```python
acc = (new_vel - state.vel) / dt if dt > _EPS else np.zeros(3)
new_pos = state.pos + state.vel * dt + 0.5 * acc * dt * dt
```
`ConstantAccel.propagate` (models.py:125-126) использует ту же формулу
интегрирования позиции (без вычисления `acc` через разность скоростей, у неё
`acc` уже известно) — итого формула кинематического шага размазана по 4
классам-стратегиям.

**Почему нарушение:** GRASP Low Coupling / High Cohesion — все 4 класса
знают детали численного интегрирования вместо того, чтобы это знал один
Information Expert. Тот же риск дрейфа формул, что и в предыдущей находке:
если понадобится, например, перейти на другую схему интегрирования (RK2 вместо
semi-implicit Euler) — придётся править 3-4 места синхронно.

**Как исправить:** вынести helper (свободная функция или метод на
`TargetState`/в `kinematics.py`, где уже есть похожий Pure Fabrication для
геометрии): `_integrate(state, new_vel, dt) -> TargetState`, возвращающий
`state.evolved(pos=..., vel=new_vel, acc=...)`. Каждая стратегия вычисляет
только свой `new_vel` (это и есть их различающаяся ответственность) и
делегирует интеграцию общему helper'у.

---

### [LOW] Python-цикл вместо векторизации — `core/snr/estimator.py:169-175`

**Что:** сбор CFAR reference-window (`ref_values`) через `for i in
range(cfg.ref_bins)` с ручным wraparound по модулю `n_fft`.

**Почему нарушение:** `05-python-style.md` явно требует «Векторизация вместо
python-циклов где возможно». `ref_bins` по умолчанию 8 (16 значений) — цена
цикла сейчас мала, но это единственное место в модуле, отступающее от
собственного правила стиля проекта.

**Как исправить (не обязательно, low prio):** можно собрать индексы
`np.arange` + `% n_fft` и взять `mag_sq[idx_left]`, `mag_sq[idx_right]` без
python-цикла, если пайплайн станет горячим местом (иначе не критично).

## Удачно

1. **`Kinematics` как Information Expert / Pure Fabrication** (`kinematics.py`) —
   геометрия апертуры (шаг решётки, длина волны, разрешение по дальности)
   изолирована от `TargetState`, конфиг инжектится через конструктор (DIP), сам
   класс не мутирует ни `state`, ни `cfg` — чистая проекция state→бины.
2. **`MotionModel` как Strategy + Registry в вызывающем коде** — единый
   Protocol-контракт `propagate(state, dt, rng) -> TargetState`, все 5
   реализаций (`ConstantVelocity/MarkovDrift/CoordinatedTurn/ConstantAccel/
   WeavingManeuver`) взаимозаменяемы и реально используются полиморфно через
   `list[MotionModel]` (`core/generators/tact_sequence.py`) и factory-словарь
   по имени (`core/runtime/commands.py:_build_motion`) — новая модель
   движения = новый класс + строка в словаре, без правки существующего кода (OCP).
3. **`StatisticsSnrEstimator` без `SnrConfig` — осознанное ISP-решение**,
   явно задокументированное в docstring («Не требует SnrConfig: спектральный
   конфиг не нужен для time-domain») — хороший пример разделения интерфейсов
   по реальной потребности, а не «на всякий случай одинаковый конструктор».

## Сводка: 1 high / 3 medium / 1 low
