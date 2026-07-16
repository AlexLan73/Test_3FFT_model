# 🧩 TASK — Body-Motion · P3 (помехи: заград + гребёнка + промышленные)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** интеграция помех в объём такта.
> **Спека:** [`SPEC.md`](../specs/body_motion_3d_2026-07-15/SPEC.md) (§7·S3), ревью F3.
> **Зависит от:** P2 (объём + шум). **Статус:** ✅ ПРИНЯТО (ревью Кодо 2026-07-15).
>
> 🚨 🚫 pytest (`TestRunner`) · 🚫 `.claude/worktrees/**` · существующее НЕ ломать · РЕЮЗ помех, не переписывать.

---

> 🧭 **Конвенция 3D-визуала (везде):** дальность (range) — по **горизонтали** (пол сцены, вдаль); kx (азимут) — вбок; ky (угол места) — вверх. Динамика: GIF + `--live` окно + `.html`.

## 🎯 Цель P3

Добавить в объём такта (P2) **помехи** поверх цели: заградительная (rank-1 с угла), гребёнка
(DRFM — задержанные ложные цели по дальности), опц. промышленные (CW/VFD/дуга/клаттер).
Включаются флагами из конфига. Реюз готовых помех, **без плодиловки**.

## 🧭 Опорные решения (спека/патент/ревью)
- **Два слоя помех уже есть (F3):** сигнал-уровень `core/generators/waveforms/jammers_rf.py`
  (6 помех P5) и `core/generators/jammers.py` (`DrfmComb`/`BarrageJammer`/`HamEmitter`).
- 🔴 **A2 — домен `jammers.py` = RAW fast-time `(nx,ny,n_real)`, ДО дальностного FFT** (проверено:
  `BarrageJammer.contribute` → `steer[:,:,None]·noise[:]` формы `(nx,ny,n_real)`). Это **совпадает** с
  входным объёмом P2 `16×16×N` → реюз **правомерен**, вставляем помеху **в raw-объём** (не в FFT-куб).
  Сигнатура `contribute(grid, rng, rs)`: `rng` = **`RangeConfig`** (не Generator!), `rs` =
  `default_rng`. Согласовать `RangeConfig.n_real == N`. **Не трогать** `jammers.py`, вызывать как есть.
- **Заград/гребёнка кладутся в raw-объём**; сигнатура «полоса во всех блоках дальности» (гл.4-бис)
  проявится **после FFT** (P5), не в P3. **Гребёнка** — задержанные копии по дальности (raw).

## 📦 Что создать / расширить

### 1. `core/generators/scene_modeler.py` — `SceneModeler` — **реюз `Scene`, НЕ второй Composite (A1)**
🔴 **A1:** `Scene`(Composite: `add`/`contribute`)/`SceneBuilder`/`Synthesizer` **уже есть**
(`core/generators/scene.py`). **НЕ плодить второй Composite.** `SceneModeler` = тонкий строитель, который
**наполняет существующий `Scene`** источниками такта (цель(и) P2 + помехи) и делегирует
`Scene.contribute(grid, rng, rs)`. Если хватает `SceneBuilder` — использовать его, `SceneModeler` не
создавать вовсе (обосновать на ревью). Реюз существующих помех как `SignalSource`:
- заград: `jammers.BarrageJammer` (raw-домен, параметры из `SceneConfig.BarrageSpec`);
- гребёнка: `jammers.DrfmComb` (`DrfmCombSpec`);
- промышленные (опц.): сигнатуры из `waveforms/jammers_rf.py` (INT_CW/IMP_ARC/VFD) + клаттер.

### 2. Флаги включения — из `ProjectConfig.scene` / `SceneConfig`
`EnableJammer{barrage,comb,cw,vfd,arc,clutter}` — читаются из конфига (в P6 те же флаги
поедут как команды панели). Дефолт-сцена: цель + заград + гребёнка.

### 3. Демо S3 — дополнить `demo_body_motion.py`
Объём с сигнатурами помех → 3D-визуал в `graphics/body_motion/p3_jammers/`:
заград — полоса на угле по всей дальности; гребёнка — цепочка пиков по Z; цель — компактный пик.

## ♻️ Реюз (точные пути)
- Куб-уровень помех: `core/generators/jammers.py` (`BarrageJammer`,`DrfmComb`,`HamEmitter`) — **как есть**.
- Сигнал-уровень: `core/generators/waveforms/jammers_rf.py` (INT_CW/IMP_ARC/VFD/SMSP…).
- Спеки помех: `core/config/scene_config.py` (`BarrageSpec`,`DrfmCombSpec`,`ThermalNoiseSpec`).
- Существующая сборка сцены: `core/generators/scene.py` (свериться, реюз, не дублировать).

## ✅ Критерии приёмки
- Заград: rank-1 сигнатура (энергия на одном угле во всех дальностных блоках) — проверка
  по срезу/ранговости. Гребёнка: ≥3 равноотстоящих ложных пика по дальности с одного угла.
- Цель поверх помех **выживает** (компактный пик отличим от полосы заграда).
- Флаги вкл/выкл каждой помехи работают; входы не мутируются.
- Куб-уровневый `jammers.py` не изменён.

## 🧪 Тесты (`TestRunner`)
`JammerSceneTests`: rank-1 заграда, число/шаг пиков гребёнки, сосуществование с целью,
флаги. Старые целы, ruff/mypy 0.

## 🚫 Границы
Без мульти-цели (P4 — здесь одна цель + помехи), без FFT/квадратов (P5), без сокета (P6).

---

## 🔎 Сверка Кодо с кодом (2026-07-15, перед реализацией) — ОБЯЗАТЕЛЬНО К УЧЁТУ

Подтверждено: A1 (`Scene`/`SceneBuilder`/`Synthesizer` в `core/generators/scene.py`;
`EmitterFactory` уже регистрирует `BarrageSpec`/`DrfmCombSpec`/`HamEmitterSpec`) и
A2 (`contribute(grid: ArrayGrid, rng: RangeConfig, rs: Generator)` → `(nx,ny,n_real)`).

- 🔴 **K1 — двойной шум.** `SceneBuilder.build()` **всегда** добавляет `ThermalNoise`
  (scene.py:41), а объём P2 уже содержит шум (`render_pipeline`, `NOISE_POWER=1.0`,
  калибровка по `snr_db`). Реюзить `SceneBuilder.build` как есть НЕЛЬЗЯ — будет два шума.
  → Строить **jammers-only `Scene`** напрямую: `Scene()` + `EmitterFactory.create(spec)`
  по включённым флагам, **без** `ThermalNoise`. `SceneBuilder` не трогать.
- 🔴 **K2 — RangeConfig под N объёма.** Дефолтный `cfg.range_` имеет `n_real=16`, а объём
  P2 — `N=VolumeBuilder.n_samples` (1024). Для вызова `contribute` создавать
  `RangeConfig(n_real=N, n_fft=N)` от фактического N объёма (см. критерий таска
  «Согласовать RangeConfig.n_real == N»). Несовпадение форм = сломанная сумма.
- 🟡 **K3 — шкала мощности помех.** Шумовой пол объёма P2 = `NOISE_POWER=1.0` (σ²),
  цель ≈ `snr_db=12`. `BarrageSpec.power=6.0` даёт ~7.8 дБ над полом — для наглядной
  «полосы» в демо мощность помехи задавать из спеки (например 20–60), НЕ хардкодить
  в коде. Критерий «цель выживает» проверять при дефолте.
- **Флаги (п.2 таска):** добавить frozen VO `JammerFlags` в `core/config/scene_config.py`
  (bools: `barrage, comb, ham, cw, vfd, arc, clutter` — дефолт все False) + поле
  `SceneConfig.jammers: JammerFlags = field(default_factory=JammerFlags)` (обратная
  совместимость — старые вызовы целы) + спеки помех в отдельных полях либо реюз
  `emitters`. Демо включает barrage+comb.
- **Интеграция с P2:** помехи суммируются с объёмом **после** `VolumeBuilder.build_from_sample`
  (`vol + jammer_scene.contribute(ArrayGrid.from_config(cfg.array), RangeConfig(n_real=N,
  n_fft=N), rs)`), вход не мутировать. Промышленные (cw/vfd/arc/clutter) — опционально,
  через адаптер сигнатур `jammers_rf` ТОЛЬКО если просто; иначе оставить заглушку-флаг
  с NotImplementedError и пометить в отчёте.
- **Запись файлов:** через bash heredoc / `python - <<'EOF'` + `ast.parse` после
  (ФС-гоча среды). Тесты: `python3 tests/all_test.py`.

---

## ✅ РЕВЬЮ КОДО (2026-07-15) — ПРИНЯТО

Реализовано Sonnet-агентом: `core/generators/scene_modeler.py` (`SceneModeler`,
Builder — реюз `Scene`, второй Composite НЕ создан, A1 ✓), `JammerFlags` +
optional-спеки в `core/config/scene_config.py`, демо `demo_body_motion_jammers.py`,
тесты `tests/test_body_motion_jammers.py`. `jammers.py`/`scene.py`/`factory.py` не трогали.

**Сверка находок:**
- **K1** ✓ — jammers-only `Scene` строится вручную по флагам, `ThermalNoise` НЕ добавляется
  (двойного шума нет; объём P2 уже калиброван по `snr_db`).
- **K2** ✓ — `RangeConfig(n_real=N, n_fft=N)` от фактического `volume.shape[2]`, не `cfg.range_`.
  Тест `range_config_matches_actual_n` проверяет N=256/1024, дефолт `cfg.range_.n_real=16` цел.
- **K3** ✓ — мощности из спек (дефолт спеки при `None`), в коде не хардкод; `cw/vfd/arc/clutter`
  → явный `NotImplementedError` (не тихий no-op).

**Проверки:** тесты **9 ok/0 fail** (весь набор зелёный); ruff/mypy — 0 замечаний;
демо пишет 4 файла в `graphics/body_motion/p3_jammers/`. Угловая карта 16×16 показывает
**три разнесённых по углу источника** (заград rank-1 · цель · гребёнка); профиль дальности
честно показывает сырой домен (разделение по дальности — задача P5 после дечирпа/FFT).
Входы не мутируются, флаги вкл/выкл работают.

**Замечаний-блокеров нет.** Мелочь (не блокер): демо при высоких демо-мощностях помех
показывает «цель в окне: False» — это ожидаемо в сыром домене (сжатие по дальности — P5);
критерий «цель выживает» тестируется на дефолтах спек и проходит.
