# 🔍 Архитектурное ревью radar3d — ООП/SOLID/GRASP/GoF (2026-07-20)

> Заказ Alex: глубокое ревью репо на соответствие ООП/SOLID/GRASP/GoF, нарушения — обсудить и исправить.
> Метод: 6 параллельных ревьюеров (Sonnet) по кластерам + синтез/верификация Кодо. Детали по кластерам —
> `MemoryBank/specs/review/<кластер>_2026-07-20.md`. **Код НЕ правился** — только находки (правки после обсуждения).

## Итог: 4 HIGH · ~13 MED · ~11 LOW

Общая оценка: **архитектура здоровая** — паттерны проекта (Strategy/Factory/Registry/VO/Composite/DI)
реально применены в большинстве мест (валидаторы, config, tokenizer, MotionModel, Transport, camera —
образцовые). Нарушения — точечные, сгруппированы в 4 повторяющиеся темы (ниже).

---

## 🔴 HIGH (перепроверено Кодо)

### H1. `TargetState` — VO протекает (не иммутабелен) · `core/motion/state.py:9-12` ✅живой баг
`_vec3()` не копирует float64-вход (`np.asarray`+`setflags(write=False)` на view). Мутация исходного
массива ПОСЛЕ конструирования меняет `state.pos` (проверено: `shares_memory=True`, 1.0→999.0).
**Фикс:** `np.array(value, dtype=np.float64).reshape(3)` (copy). Мелкий, безопасный.

### H2. `_scene_to_dict` теряет помехи при сериализации · `core/data_context/run_workspace.py:61-65` ✅подтв.
Отдаёт только `emitters`+`thermal`; `SceneConfig` имеет ещё `jammers`/`barrage_spec`/`comb_spec`/`ham_spec`
(scene_config.py:96+) — они молча теряются в manifest.yaml. Риск воспроизводимости.
**Фикс:** дописать поля в сериализацию (+ парная десериализация). **Обсудить:** полагается ли перезагрузка
на manifest или регенерит по seed (тогда severity ниже).

### H3. Нет абстракции «nuller» — `AntiBarragePipeline` завязан на конкретику · `core/models/anti_barrage/` ✅подтв.
`pipeline.py:44` типизирован `nuller: SubspaceNuller`; `RobustMvdrNuller` (mvdr.py) реализует тот же
контракт `apply()`, но общего ABC/Protocol нет и MVDR нигде не подключён (только тест). Единственное место
семейства без Strategy+DI (у RadarModel/Classifier/Arbiter/Tracker он есть).
**Фикс:** ввести `Nuller(Protocol/ABC)` с `apply()`, типизировать pipeline им (DIP/OCP).

### H4. `HipBackend` не взаимозаменяем с `NumpyBackend` (LSP) · `core/generators/backends/hip_backend.py:33-86`
`Waveform.render(backend,...)` использует `backend` только для окна/шума — несущую считает CPU-код;
`HipBackend` в этом контракте даёт CPU-идентичный выход, реальный GPU-путь — отдельный несовместимый метод.
**⚠️ Касается GPU:** согласовать с GPU-чатом (это их территория backends) — возможно, часть их текущей работы.

---

## 🟡 MED (по темам — детали в файлах кластеров)

- **Дубль математики (рег. 07 «единый источник»):** ковариация+diagonal-loading в `nuller.py` vs `mvdr.py`;
  формула semi-implicit интегрирования в `MarkovDrift`/`CoordinatedTurn`/`WeavingManeuver` (motion/models.py);
  пайплайн decimation→window→pad→FFT в `SpectrumSnrEstimator.estimate` vs `.get_mag_sq` (snr).
- **Дубль подписей осей:** `angular_map.py:30-31` хардкодит `"kx (азимут)"/"ky (угол места)"` = дубль
  `layout.py:_LABELS` (рег. 06 — «подписи только в AxisLayout», тот же класс, что прецедент 07).
- **VO не frozen:** `TestResult`/`ValidationResult` (`common/result.py`) — `@dataclass` без `frozen`, но
  докстринг зовёт их «неизменяемыми VO»; `.add()` мутирует и возвращает self.
- **DIP не дотянут:** `NumpyBackend()` хардкодится (не инжектится) в `VolumeBuilder`/`waveform_to_cube` —
  GPU-путь недостижим без правки этих файлов, хотя `GenBackend` для того и есть.
- **LSP (snr):** `StatisticsSnrEstimator.estimate` ужесточает предусловие (ValueError при `support=None`),
  не взаимозаменяем с `SpectrumSnrEstimator` по общему Protocol.
- **Strategy на словах:** `ScenePointsVisualizer` не наследует `Visualizer(ABC)`, сигнатура `render` иная.
- **ISP/LSP (runtime):** `Transport` Protocol связывает publish+subscribe; `WebSocketTransport.subscribe`
  кидает NotImplementedError → разделить на `PublishTransport`/`SubscribeTransport`.
- **SRP (runtime):** `SceneServer` совмещает кинематику+объём+токенизацию+сериализацию+publish — вынести
  `_*_payload` в Pure Fabrication-сериализатор.
- **Стиль:** `data_context/repository.py` на `os.path` вместо pathlib (+ нет type hints у `NpyCubeRepository`);
  `print()` в либе `common/gpu_context.py:77`; dtype `complex128` в `sources.py:25._empty` vs `complex64` всюду.

## 🟢 LOW (мелочи, детали в файлах)
Type hints на override'ах `SignalSource.contribute` отсутствуют; магическая строка colorbar
(`scene_points.py:91`); python-цикл вместо векторизации (snr CFAR ref-window); абсолютный импорт в
`anti_barrage/cfar.py:26`; `RawQueue.get` одиночный wait (spurious wakeup); `WebSocketTransport.publish`
ленивый side-effect start; stringly-typed `decision`/`kind` без `Literal`.

---

## 🎯 4 сквозные темы (корень большинства находок)

1. **Иммутабельность VO не гарантирована** — `TargetState` (H1), `result.py` (MED). Паттерн-фикс:
   `frozen=True` + копирование входных массивов в конструкторе.
2. **DRY / единый источник математики (рег. 07)** — 4 дубля формул (cov+loading, интегрирование×3, FFT-пайплайн,
   подписи осей). Это ровно класс риска из прецедента 07 — вынести в один хелпер каждую.
3. **Strategy/DIP объявлены, но не дотянуты** — nuller без ABC (H3), backends хардкод (MED), HipBackend не
   взаимозаменяем (H4). Абстракция есть — не подключена/не типизирована.
4. **Наминг/стиль-нитки** — os.path vs pathlib, print в либе, type hints на override, dtype complex128.

## Статус правок (решения Alex приняты, порядок 4→2→1→3)

### ✅ Волны 1+2 СДЕЛАНЫ (2026-07-20, behavior-preserving, все тесты зелёные, ревью Кодо)
- **H1** ✅ `TargetState` copy (`np.array` вместо `np.asarray` view) — не течёт (`shares_memory=False`).
- **H2** ✅ `_scene_to_dict`/`from_dict` сериализуют `jammers`/`barrage`/`comb`/`ham_spec` обе стороны
  (+ `tests/test_run_workspace.py` round-trip dict/yaml/defaults, 3 ok).
- **H3** ✅ `Nuller(Protocol)` (`base.py`) — `AntiBarragePipeline` типизирован абстракцией; MVDR не подключён.
- **Дубли математики (рег.07)** ✅ единый источник: `_covariance.py` (cov+diagonal-loading — nuller/mvdr),
  `_integrate_state` (motion ×3), `SpectrumSnrEstimator._spectrum` (snr).
- **Стиль** ✅ `print`→`warnings` (gpu_context), подписи осей → `AxisLayout` (angular_map).

**⚠️ Судейские решения (на обсуждение):**
1. `result.py` оставлен МУТАБЕЛЬНЫМ (вариант «б»): вендор-код, `.add()` нигде не вызывается — поправлен
   только докстринг (убрано «неизменяемый»), а не введён `frozen`. Если хочешь строгий frozen — скажи.
2. `ConstantAccel` НЕ через `_integrate_state` (считает acc напрямую, роутинг дал бы деление на dt=0).
3. **Nuller LSP не полный:** `SubspaceNuller.apply`→куб, `RobustMvdrNuller.apply`→луч (разные формы).
   OCP/DIP улучшены (pipeline на абстракции), но истинной взаимозаменяемости нет. При желании — общий
   контракт формы выхода (отдельная задача).

### ⏳ Волна 3 (с GPU-чатом) — отложена
- **H4** `HipBackend` не LSP-substitutable, **DIP backends** хардкод `NumpyBackend`, **dtype** complex128
  в `sources.py` — всё в `core/generators/**` (территория GPU-чата). Согласовать, чтобы не конфликтнуть.

### Не берём (по решению Alex)
- `Transport` split (ISP) — оставлен рабочий обход (FanOut ловит NotImplementedError).
- `SceneServer` сериализатор — не в скоупе (MED, низкая срочность).

### ✅ Хвост MED/LOW — безопасная часть СДЕЛАНА (2026-07-20, коммит `3c73454`)
- ✅ `anti_barrage/cfar.py` абсолютный импорт → относительный.
- ✅ snr LSP: `SnrEstimator` Protocol — честная документация предусловия `support` (ValueError у
  `StatisticsSnrEstimator` математически обоснован — вариант «а»).
- ✅ `scene_points.py` colorbar: явный параметр `render(colorbar: bool|None)` вместо магической строки gid.
- ✅ `RawQueue.get`: цикл по монотонному дедлайну (защита от spurious wakeup).

**Осталось (по желанию / зависит):**
- `ScenePointsVisualizer` не наследует `Visualizer(ABC)` — сигнатура `render` иная (points vs cube);
  «фикс» = общая абстракция или документирование. Дизайн-вопрос, не мелочь — обсудить при желании.
- `repository.py` os.path→pathlib + type hints — в `core/data_context/**`; безопасно, но отдельно.
- stringly-typed `decision`/`kind` → `Literal` (tokenizer/arbiter) — осознанный выбор ради сериализации,
  трогает models — отложено.
- type hints на `SignalSource.contribute` override — в `core/generators/**` → Волна 3 (GPU-чат).
