# Ревью: generators

Кластер: `core/generators/**` (incl. `waveforms/**`: mseq, waveform_to_cube, jammers_rf;
scene_modeler, volume, tact_sequence и остальные `.py`).

## Находки (severity ↓)

### [HIGH] LSP — `HipBackend` НЕ взаимозаменяем с `NumpyBackend` через контракт `Waveform.render` — `core/generators/backends/hip_backend.py:33-86`

**Что:** Докстринг `core/generators/waveforms/base.py:8-11` и `hip_backend.py:8-11`
заявляют: «Две реализации (DIP/LSP, взаимозаменяемы): `NumpyBackend` / `HipBackend`».
Но каждый `Waveform`-подкласс (`cw.py:20-23`, `lfm.py:26-32`, `am.py:37-42` и т.д.)
считает «сырой» сигнал шага 1 **сам, через `reference.cw_numpy`/`getX_numpy` (чистый
CPU numpy)**, и передаёт готовый `signal` в `render_pipeline(backend, ...)` — параметр
`backend` там используется ТОЛЬКО для `apply_window`/`add_noise` (шаги 2/4), никогда
для генерации несущей. `HipBackend._generate_raw`/`.render(modulation, spec, rng)`
(строки 63-86) — это ОТДЕЛЬНЫЙ метод с ДРУГОЙ сигнатурой (`(modulation, spec, rng)`,
не входит в `GenBackend`-протокол `base.py`), который единственный реально считает
несущую на GPU.

Подставив `HipBackend()` в `SomeWaveform().render(backend, spec, rng)` (ровно то
использование, которое подразумевает Strategy/LSP-контракт `GenBackend`), GPU-код
`_generate_raw`/`gen.generate()` **не вызывается вообще** — результат тождественен
`NumpyBackend`, только молча (без исключения). Единственный способ реально получить
GPU-синтез — вызвать `HipBackend().render(modulation, spec, rng)` НАПРЯМУЮ, в обход
`Waveform`/`WaveformFactory`. Это подтверждено самим тестом
`tests/test_generators.py:403-404`, где для сравнения используются ДВА РАЗНЫХ вызова:
`hip.render(Modulation.CW, spec, rng)` против `CwWaveform().render(numpy_backend, spec, rng)`
— т.е. даже автор тестов не мог использовать единый интерфейс.

**Почему нарушение:** Substitutability (LSP) — главное обещание Strategy/DIP здесь:
«клиентский код (`Waveform.render`) не отличает бэкенды». По факту клиентский код
(`Waveform`-подклассы) НИКОГДА не пользуется GPU-веткой `HipBackend`, а прямой вызов
`HipBackend.render(...)` — это второй, несовместимый интерфейс, который ломает
подстановочность без единого сообщения об ошибке.

**Как исправить:** Либо (а) вынести генерацию шага 1 в сам `GenBackend`-протокол
(`generate_raw(modulation, spec) -> np.ndarray`), и `Waveform.render` вызывает
`backend.generate_raw(...)` вместо `reference.cw_numpy`/`getX_numpy` напрямую — тогда
подмена бэкенда реально меняет поведение; либо (б) явно задокументировать/переименовать,
что `HipBackend` НЕ реализует `GenBackend` для боевого пути, а является отдельным
Facade поверх GPU-генератора, и убрать формулировку «взаимозаменяемы (LSP)» из докстрингов.

### [MED] DIP — `NumpyBackend` захардкожен внутри «боевого» тракта построения объёма, GPU-бэкенд недостижим — `core/generators/volume.py:107,128`, `core/generators/waveforms/waveform_to_cube.py:99,150,168`

**Что:** `VolumeBuilder.build_from_sample` (единственная точка входа для наполнения
куба такта, `volume.py:79-108`) сама создаёт `NumpyBackend()` внутри метода
(`field = waveform.render(NumpyBackend(), spec, rng)`, строка 107) и не принимает
бэкенд как параметр/поле конструктора. То же в `add_shared_noise` (128) и во всех
трёх местах `waveform_to_cube.py` (99, 150, 168) — `build_lfm_target_volume`,
`build_pulse_echo_volume`, `LfmToCube`/`AmToCube` тоже не берут `backend` снаружи.
Единственные вызовы `HipBackend()` во всём репозитории — `demo_generators.py:206` и
тест `test_generators.py:394`; ни один продакшн-путь (`main.py`, `demo_body_motion_*`,
`core.controller`) до `HipBackend` не дотягивается.

**Почему нарушение:** Проект декларирует `GenBackend` как Strategy именно ради
подмены CPU↔GPU (DIP, Composition Root в `main.py`/демо). Хардкод конкретного класса
внутри `Pure Fabrication`/`Builder` (`VolumeBuilder`) убирает саму возможность
инъекции — весь трек «движение цели → куб» (P1-P5, реальный производственный тракт,
судя по последним коммитам про веб-панель полёта) физически не может использовать GPU
без правки кода этих файлов, хотя абстракция для этого существует и оттестирована.

**Как исправить:** Добавить `backend: GenBackend = field(default_factory=NumpyBackend)`
в `VolumeBuilder` (frozen dataclass — поле уже укладывается в стиль) и передавать его
явно в `WaveformToCube`-функции параметром, а не создавать `NumpyBackend()` внутри.

### [MED] NumPy dtype — `SignalSource._empty` возвращает `complex128` вместо `complex64`, расходится с остальным кластером — `core/generators/sources.py:25`

**Что:** `np.zeros((grid.nx, grid.ny, rng.n_real), dtype=complex)` — `dtype=complex`
(python `complex` → `np.complex128`). Все источники (`PointTarget`, `ThermalNoise`,
`DrfmComb`, `BarrageJammer`, `HamEmitter`) складываются в этот аккумулятор
(`Scene.contribute`, `scene.py:24-28`), и numpy promotion удерживает `complex128` до
конца цепочки. Подтверждено использованием: `demo_cfar.py:88` и `demo_nuller.py:89`
комментируют результат `Synthesizer.build()` буквально как `# (16, 16, 16) complex128`.
Этот же путь (`SceneBuilder`/`Synthesizer`) — основной генератор обучающих кубов для
3D-CNN (`core/models/classification/dataset.py:64-66`, `CubeDatasetGenerator.sample`),
т.е. каждый обучающий сэмпл идёт через complex128 FFT прежде, чем схлопнуться в
`float32` магнитуду.

**Почему нарушение:** `.claude/rules/05-python-style.md`: «Явный `dtype`
(`np.float32`/`np.complex64`) — экономия памяти под GPU-перенос». `core/generators/
waveforms/**` (более новый, P0+) последовательно использует `complex64` везде
(`_pipeline.py:79`, `field.py:70-71` даже валидирует `dtype == complex64` в
`__post_init__`) — а параллельный «сырой» тракт `core/generators/sources.py`/`scene.py`
(более старый, P0-baseline) остался на `complex128`. Расхождение конвенции внутри
одного кластера — 2× память/FFT-время без выгоды, и потенциальный сюрприз при
попытке скормить эти данные GPU-коду, ожидающему `complex64` (как в `waveforms/`).

**Как исправить:** `dtype=np.complex64` в `sources.py:25`; проверить, что все
`contribute()`-реализации (`np.exp(1j*...)`, `rs.standard_normal(...)+1j*...`) явно
кастуют в `complex64` перед сложением (сейчас — нет, см. следующую находку).

### [LOW] Типизация — `contribute()` без аннотаций во ВСЕХ реализациях `SignalSource` — `core/generators/sources.py:54,65`, `core/generators/jammers.py:21,44,61`, `core/generators/scene.py:24`

**Что:** Абстрактный метод `SignalSource.contribute(self, grid: ArrayGrid, rng:
RangeConfig, rs: np.random.Generator) -> np.ndarray` (sources.py:18-21) полностью
типизирован. Ни одна конкретная реализация (`PointTarget`, `ThermalNoise`, `DrfmComb`,
`BarrageJammer`, `HamEmitter`, `Scene`) не повторяет аннотации — везде
`def contribute(self, grid, rng, rs):`.

**Почему нарушение:** `.claude/rules/05-python-style.md`: «Type hints везде». Не
критично для рантайма, но систематично по всему кластеру (6 из 6 переопределений) —
`mypy core/` не поймает несовпадение сигнатур с ABC на этих классах.

**Как исправить:** Скопировать сигнатуру из `SignalSource.contribute` в каждый
override (или включить `--disallow-untyped-defs` для `core/generators` в CI, чтобы
не давать регрессировать дальше).

## Удачно

- **`EmitterFactory`/`WaveformFactory`** (`factory.py`, `waveforms/factory.py`) —
  чистый Abstract Factory + registry: `register(type, builder)`/`create(spec)`,
  `KeyError → ValueError` с понятным сообщением. Новая помеха/модуляция = регистрация
  одной строкой, тракт (`Scene`/`SceneBuilder`/`render_pipeline`) не трогается —
  ровно то, что требует `05-python-style.md` (OCP-таблица).
- **`WaveformSpec`/`SignalField`** (`waveforms/base.py`, `waveforms/field.py`) —
  образцовые Value Object: `frozen=True`, `meta` защищена `MappingProxyType` в
  `__post_init__` (не мутируется даже через `frozen`-обход), `SignalField.__post_init__`
  валидирует и форму осей, и `dtype`.
- **`SceneModeler`** (`scene_modeler.py`) — включение нереализованной помехи
  (`cw`/`vfd`/`arc`/`clutter`) даёт явный `NotImplementedError` со списком причин
  (строки 77-83), а не тихий no-op — хороший «fail loud» паттерн, плюс докстринг честно
  объясняет, почему `ThermalNoise` НЕ добавляется повторно (избежание удвоения шума, К1).

## Сводка: 1 high / 2 med / 1 low
