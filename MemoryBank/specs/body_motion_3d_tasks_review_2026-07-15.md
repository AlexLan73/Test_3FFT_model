# 🔬 Глубокий анализ тасков P6 «Движение тела в 3D-кубе» — 2026-07-15

> Ревью **6 тасков** `TASK_body_motion_p{1..6}` против: финализированной спеки (патент гл.3+4-бис),
> **реального кода** `core/**` и готовой базы генераторов P0–P5.
> Метод: прочитаны все 6 тасков + `jammers.py`, `scene.py`, `run_workspace.py`, `config_source.py`,
> `fft3d.py`, `data_context.py`, `array_config.py`, `numpy_backend.py`, `placement.py`.
> **Вердикт:** таски написаны грамотно и **уже учитывают патентную схему** (P5 строго по патенту,
> F1–F9 разнесены по фазам). Найдено **8 находок** (2 🔴 / 3 🟠 / 3 🟡) — не блокеры, но уточнить
> до старта Sonnet-кода.

---

## ✅ Что в тасках хорошо (подтверждено)

- **P5 — строго по патенту:** ЛЧМ = 2 раздельных FFT **без окна** (глобальный дальностный + угловой
  nx×ny→N_pad поячеечно), АМ = локальный 3D-FFT по окну `nx×ny×D` шаг 8/16/32/64. F4/F5/F6 закрыты. ⛔-блок
  «нет окна у ЛЧМ» на месте.
- **Разнесение ревью-находок по фазам:** F1/F2/F9→P1, F3→P3, F8→P6, F4/F5/F6→P5. Аккуратно.
- **Реюз готовой базы:** P2 реюзит `sources`/`grid`, P3 — `jammers.py`/`jammers_rf.py`, P5 — `fft3d`+
  `placement`, P6 — `PyPanelAntennas`. Везде 🚫 pytest / 🚫 worktrees / «не плодить».
- **Границы фаз** (что вне) прописаны в каждом таске — хорошо для Sonnet.

---

## 🔴 Критические (уточнить до кода)

### A1 — дублирование Composite: `Scene`/`SceneBuilder` УЖЕ есть, P3/P4 плодят `SceneModeler`
`core/generators/scene.py` содержит **готовый** `Scene(SignalSource)` (Composite: `add`/`contribute`),
`SceneBuilder` (из `SceneConfig` через `EmitterFactory`), `Synthesizer` (куб). Таски P3/P4 создают
**новый** `core/generators/scene_modeler.py` «SceneModeler (Builder+Composite)» — это **второй Composite**
рядом с существующим (прямое нарушение «не плодить сущности»).

**Фикс:** `SceneModeler`/`TactSequence` должны **реюзать `Scene` как контейнер** источников (Composite
уже готов) и добавлять **только** движение по тактам. Либо `TactSequence` строит новый `Scene` на такт
через `SceneBuilder`, либо оборачивает его. НЕ переписывать Composite. Уточнить в P3 §1 и P4 §1.

### A2 — домен помех: реюз **корректен** (raw), но формулировка в P3 путаная
Проверено: `jammers.BarrageJammer/DrfmComb/HamEmitter.contribute(grid, rng, rs)` возвращают массив
**`(nx, ny, n_real)` — это RAW fast-time, ДО дальностного FFT** (не «после-дечирп куб», как ошибочно
писал старый signal-gen P5). Значит домен **совпадает** с входным объёмом body_motion `nx×ny×N` →
реюз в P3 **правомерен**. НО:
- P3 пишет «заград в кубе (гл.4-бис) → выброс во всех блоках дальности» — это сигнатура **после FFT**
  (P5), а вставляем-то в **raw** (до FFT). Уточнить: помеха кладётся в raw-объём, «полоса по дальности»
  проявится после FFT в P5.
- Сигнатура `contribute(grid, rng, rs)`: `rng` здесь = **`RangeConfig`** (не Generator!), `rs` =
  `np.random.default_rng`. Нужно `rng.n_real == N` (согласовать `RangeConfig.n_real` с длиной объёма).
  Сцена/`Synthesizer` уже дают этот контракт — реюзать его, а не изобретать.

### A9 — 🔴 body_motion НЕ реюзит готовые Python-генераторы `waveforms/` (вопрос Alex)
Вчера сделан **полный слой генерации** (numpy, Windows, 48+ тестов): `WaveformFactory.create(Modulation)
→ Waveform.render(backend, spec, rng) → SignalField`. `render` уже делает **ровно** то, что P2
изобретает в `VolumeBuilder` вручную:
```
1D-сигнал (CW/ЛЧМ/АМ/ФМн/ЧМ, reference.*) → окно (TimeWindow) → раскладка n×n через grid.steering(kx,ky)
   → шум по snr_db → SignalField.data[nx,ny,N]   ← это и есть raw-вход фронтенда body_motion!
```
Готовы: `CwWaveform`,`LfmWaveform`(getX центр.чирп),`AmWaveform`,`PhaseCodeWaveform`,`FmInterference` +
6 помех; `NumpyBackend`(Windows) и `HipBackend`(GPU) — **тот же `SignalField`** (LSP).

**P2 `VolumeBuilder` дублирует `render_pipeline`** (свой `grid.steering`+`PointTarget`+`add_noise`) вместо
`WaveformFactory.create(cfg.modulation).render(spec, NumpyBackend(), rng)`.

**Фикс (внесён в P2):** `VolumeBuilder` **реюзит `WaveformFactory`+`render`+`SignalField`+`NumpyBackend`**.
Движение = на каждый такт свой `WaveformSpec` с `meta={kx,ky}` из `Kinematics`. Это:
- **решает A4 автоматом** — ЛЧМ=`LfmWaveform`, АМ=`AmWaveform`, физика зондов разная из коробки;
- даёт **GPU-путь бесплатно** (`HipBackend` — Linux, тот же контракт);
- наследует **48+ тестов** генераторов; не пишем splat/шум заново.

**⚠️ 2 gap'а (проверено кодом):**
1. **numpy-путь НЕ применяет `tau_s`** (задержка/дальность): `render_pipeline` его игнорирует; `tau_s`
   учитывает **только** `HipBackend.render` (`tau_base=spec.tau_s`, `hip_backend.py:72`). → На **Windows**
   позиция цели по дальности `R` через генераторы **не сдвигается**. body_motion реализует её сам:
   `TimeWindow(short, t0=2R/c)` (реюз `placement`, уже есть) либо tau-сдвиг по fast-time. Отметить в P2.
2. **Дечирпа нет** (`grep`: только `cw_numpy`/`getX_numpy`, нет `dechirp_numpy`). ЛЧМ-ветка P5 (дальностный
   FFT) требует дечирп = `raw · conj(getX_ref)`. → Добавить в P5 (маленькая функция, опорный = тот же `getX`).

---

## 🟠 Существенные

### A3 — `TactSequence` рождается неоднозначно (P1 заглушка → P2 «расширить» → P4 файл)
P1 демо гоняет «`TactSequence`-заглушку», P2 «расширить `TactSequence` (из P1/scene)», P4 создаёт
`core/generators/tact_sequence.py`. Класса пока **нет** (в `scene.py` только `Scene`/`Synthesizer`, без
итератора тактов). P1/P2-демо ссылаются на несуществующий класс.

**Фикс:** создать **минимальный `TactSequence`** (Iterator над `Scene`+`MotionModel`) уже в **P1**,
расширять в P2/P4. Явно указать файл рождения.

### A4 — P2 «единый вход nx×ny×N для обеих веток» — физически входы РАЗНЫЕ
Патент: ЛЧМ-вход — **дечирпованный beat-тон** (частота ∝ дальность, гл.3.1); АМ-вход — «**отдельный
лёгкий амплитудно-модулированный зонд**» (гл.4-бис.2). Это **два разных сигнала**, а не «один вход,
разная обработка». P2 генерит один raw-объём и заявляет общий вход для обеих веток — при переходе к P5
фронтенды ждут **разное сырьё** (ЛЧМ: тон на всю Z; АМ: огибающая с редкими выбросами).

**Фикс:** либо `VolumeBuilder` генерит raw под конкретную ветку (`modulation` из `ProjectConfig`), либо
**явно** пометить «прототип-этап: точка splat-ится одинаково, физика двух зондов упрощена, расхождение
закрываем в P5/при обучении». Не молчаливое «вход один» — это укусит на P5.

### A5 — P1 YAML для `ProjectConfig`: `YamlConfigSource` его НЕ покрывает
Проверено: `YamlConfigSource.load()` (а) **требует PyYAML** (`import yaml`) — в офлайн-среде его **нет**;
(б) грузит только `WaveTimeConfig`, **не `ProjectConfig`**. Без зависимостей есть **`run_workspace.py`:
`to_yaml`/`from_yaml`/`config_to_dict`** (свой дампер, dict). P1 пишет «через существующий
`YamlConfigSource`/`from_yaml`» — **смешивает два разных механизма** (`YamlConfigSource`≠свободная
`from_yaml`).

**Фикс:** для `ProjectConfig` YAML — через **`run_workspace.to_yaml/from_yaml`** (без PyYAML) +
расширить `config_to_dict` на `ProjectConfig`. `YamlConfigSource`(PyYAML) — опционально, не основной путь.

---

## 🟡 Уточнения

### A6 — P5 «реюз углового ядра `fft3d`» неточно
`Fft3DModel._transform` = **полный 3D `fftn`** (`s=(nx,ny,n_fft)`) + fftshift по угловым. Для **АМ**
(локальный 3D-FFT по окну) — подходит **как есть** ✓. Для **ЛЧМ** нужен раздельно: глобальный
дальностный FFT по Z + **угловой 2D-FFT поячеечно** — такого метода `Fft3DModel` **не экспонирует**.
**Фикс:** для ЛЧМ угловой FFT = `np.fft.fftn(axes=(0,1))` поячеечно (новый код или рефактор `Fft3DModel`
с выделением углового шага), **не** вызов `Fft3DModel.process` целиком. Уточнить P5 §2.

### A7 — P6: граница `MessageBus`(in-process, P1) ↔ `Transport`(ZMQ, inter-process)
Таск смешивает «панель — Observer шины `MessageBus`» и «приём через `Transport` ZMQ SUB». В **разных
процессах** `MessageBus` (P1, внутрипроцессный) не работает. **Фикс:** явно описать мост — продюсер →
ZMQ → (сторона панели) ZMQ-приёмник кладёт в **локальный** `MessageBus`/queue → GUI-Observer. На каждой
стороне свой `MessageBus`; `Transport` — межпроцессный мост между ними.

### A8 — `IN_PROGRESS.md`: дубликат блока P6
Два блока про P6: строки 7–22 (новый, с таблицей тасков) и 78–88 (прежний мой). Свести в один
(таблица тасков + next). **← правится сразу этой сессией.**

---

## 📋 Матрица реюза (что готовая база даёт P6)

| Нужно P6 | Готовое (проверено) | Файл |
|----------|---------------------|------|
| Composite сцены | `Scene`/`SceneBuilder`/`Synthesizer` | `generators/scene.py` |
| Помехи raw-домена | `BarrageJammer`/`DrfmComb`/`HamEmitter` (nx,ny,n_real) | `generators/jammers.py` |
| Помехи сигнал-уровня | 6 шт (BarrageRF/SMSP/DRFM/INT_CW/IMP_ARC/VFD) | `generators/waveforms/jammers_rf.py` |
| Стиринг/раскладка | `ArrayGrid.steering`, `PointTarget`, `ThermalNoise` | `generators/{grid,sources}.py` |
| Окно (только АМ) | `TimeWindow` (full/partial/short) | `generators/waveforms/placement.py` |
| FFT-ядро (АМ 3D-FFT) | `Fft3DModel.fftn` (3D) | `models/fft3d.py` |
| YAML без зависимостей | `to_yaml`/`from_yaml`/`config_to_dict` | `data_context/run_workspace.py` |
| Facade данных | `DataContext` (save/load — расширяем) | `data_context/data_context.py` |
| 3D-визуал | plotly interactive, `FigureWriter`/`HtmlWriter` | `graphics/` |
| Панель-**образец** (пишем своё, не копия) | `color_map`/`geometry`/`data_models` (⚠️ UDP+JSON) | `E:\C++\GPUWorkLib\PyPanelAntennas\Examples\` |

**Новое (нет в базе, создаётся P6):** `ProjectConfig`, `MessageBus`, `core/motion/*`, `TactSequence`,
`WaveformToCube`(`LfmToCube`/`AmToCube`), объёмный токенизатор (этап детектора), `core/runtime/*`.

---

## 🎯 Резюме для Alex

Таски **готовы к реализации**, схема FFT в них верна (патент).

### ✅ Все 9 находок ВНЕСЕНЫ в таски (2026-07-15, «сразу правь ошибки»)
- **A9** (🔴 вопрос Alex про Python-генераторы) → P2 §1 + ♻️: `VolumeBuilder` реюзит
  `WaveformFactory`+`Waveform.render`+`SignalField`+`NumpyBackend` (вчерашний слой, 48+ тестов), не
  изобретает splat. Gap1 (numpy не применяет `tau_s` → дальность окном `TimeWindow`) → P2. Gap2 (дечирпа
  нет) → P5 §0. Побочно закрывает A4 (ЛЧМ=`LfmWaveform`/АМ=`AmWaveform`, физика из коробки).
- **A1** → P3 §1 + P4 §1: реюз существующего `Scene`-Composite, второй не плодим.
- **A2** → P3 опорные решения: домен `jammers.py` = raw `(nx,ny,n_real)`, сигнатура `contribute(grid,
  RangeConfig, rs)`, `n_real==N`.
- **A3** → P1 §5: `TactSequence` создаётся минимальным в P1; P2 §2 / P4 §2 — «расширить».
- **A4** → P2 опорные + §1: `VolumeBuilder` ветвится по `cfg.modulation`; физика зондов расходится в P5.
- **A5** → P1 §1: YAML `ProjectConfig` через `run_workspace.to_yaml/from_yaml` + `config_to_dict`, не
  `YamlConfigSource`.
- **A6** → P5 §2: ЛЧМ угловой FFT = `np.fft.fftn(axes=(0,1))` поячеечно + глоб. дальностный по Z; не
  весь `Fft3DModel`.
- **A7** → P6 опорные: `MessageBus` (in-process) vs `Transport` (ZMQ inter-process) — граница описана.
- **A8** → `IN_PROGRESS.md`: дубликат блока P6 сведён в один.

Старт — **S1/P1** (ProjectConfig+шина+motion), таск P1 уже содержит правки A1/A3/A5.
