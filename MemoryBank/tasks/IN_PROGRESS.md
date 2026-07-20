# 🔄 IN_PROGRESS

> Короткий указатель на активную задачу (1–5 строк). Детали — в `TASK_<topic>_<phase>.md`.
>
> 🤝 **Передача дел (2026-07-15, продолжать на Linux):** `sessions/2026-07-15_HANDOFF_P3P4_to_P6.md`.
> P1–P5 ✅, P3+P4 приняты в этой сессии. Остался **P6** (сокет-панель). Не закоммичено (Alex пушит).
> ⚠️ Перед git: `rm -f .git/index.lock _sync_test.txt`.

## Сейчас в работе

- 🎯 **АКТИВНАЯ: реалтайм-панель, ЭТАП A** (2026-07-20) — таск `tasks/TASK_realtime_panel_A.md`,
  спека `specs/realtime_panel_2026-07-19.md` (§1–§4 СОГЛАСОВАНЫ, коммит `d187639`).
  Этап A (БЕЗ GPU) делегирован Sonnet: `core/runtime/panel_publisher.py` (`Tick`/`TickLog`/
  `PanelPublisher`) + `demo/ex4_flight/live_demo.py` + тесты. Кодо ждёт → глубокое ревью + сама гонит тесты.
  ⚠️ **Два чата:** этот = база/demo (Этап A), ДРУГОЙ = GPU-обработка сырья. `SceneServer` не трогаю.
  Дальше: Этап B (сырой куб→фронтенд по `sig`→ядро→tick), хук `on_connect` (§4.4).

- ✅ **Web-панель полёта НА СОКЕТАХ (P6)** (2026-07-19) — СДЕЛАНО. `demo/ex4_flight/server.py`
  (гоняет `Ex4Flight.run_history` раз → стримит такты `meta`/`tick` через `WebSocketTransport`,
  порт 8765, дефолт 90 тактов) → тонкий клиент `web/` (`index.html`+`app.js` canvas-рендер канона
  + свой `msgpack.js` декодер, ноль внешних URL, офлайн). Проекция на клиенте единой
  `core.graphics.Projection` (`as_js()`). **Правки Alex 2026-07-19:** (1) правое поле = вид
  С НУЛЕВОЙ дальности — `Projection.field()` `az=π` (+kx вправо, наблюдатель РЛС); 3D-куб свой
  ракурс. (2) дефолт 90 кадров. (3) сквозные номера track_id в 3D·поле·теплокартах·таблице
  (в поле разведены anti-overlap). Тесты: `test_camera` 6 ok, `demo/tests/test_ex4_server.py`
  6 ok, `web/tests/test_msgpack.mjs` (node) ✓, **demo-агрегатор ВСЁ ЗЕЛЁНОЕ (10 наборов)**.
  Старый самодостаточный `web.py` → `архив/ex4_web_selfcontained_2026-07-19.zip` (удалён из дерева).
  **GPU-генерация — отложена** (Alex: CPU сейчас; torch-ROCm на его Debian+RX9070, дома нет torch).
  НЕ закоммичено (за Alex). Запуск: `server.py` → открыть `web/index.html`.

- ✅ **Web-панель полёта по канону ex4** (2026-07-19) — Alex: v1-прототип (ex5-web) красивый,
  оставить, но канон = компоновка ex4 §1.6 (3D сцена · поле с хвостами K=8 · тепловые срезы
  + ТАБЛИЦА признаков §4.11 — «кусочек с таблицей используем»). Спека
  [`specs/web_panel_flight_2026-07-19.md`](../specs/web_panel_flight_2026-07-19.md);
  решения Alex: **1A** полный 64×64×4096×30 (GPU — позже), **2A** кроп срезов ±8,
  **3B** парабола в тракт — отдельной задачей.
  **Ключевое: web = ТРЕТИЙ рендерер существующей history Ex4Flight** (R5 — не пересчитывать):
  рефактор `Ex4Flight.run_history(rng)` (вынесен из `visualize`, поведение GIF/PNG не тронуто) →
  `demo/ex4_flight/web.py`: `history_to_json` (кроп срезов, финальные признаки) + самодостаточный
  HTML+JS. Страница: ① 3D canvas с ВРАЩЕНИЕМ мышью/зумом (детекции=turbo-дБ, истина-треугольники,
  barrage-линия, хвосты) · ② поле с хвостами K=8, №+✈, квадрат носителя, рамка barrage (формула
  альфы из `_compose_frame` буквально) · ③ срезы-теплокарты ±8 + HTML-таблица признаков
  (тултипы-описания §4.11, вердикт из is_moving) · контролы + deep-link `#t=N` (пауза).
  Смотрено глазами headless-Edge (такт 1 и такт 6: всё рисуется, треки ЛЕТИТ, полоса→null).
  Ex4WebTests 4 ok (сериализация/кроп/6 признаков/самодостаточность), **demo агрегатор
  10 наборов ВСЁ ЗЕЛЁНОЕ** (ex4 после рефактора 6 ok). НЕ закоммичено.
  Следующее (по 3B): парабола `refine_peak` в тракт после `coarse_burst_points` — треки суб-биновые.

- ✅ **Параболическое уточнение пика в кубе (ex5)** (2026-07-19) — решение Alex по
  TASK_ex1_search_p2 §3: парабола (не Бессель). Core: `core/models/tokenizer/peak_refine.py`
  (`refine_peak` — лог-парабола δ=½(P₋−P₊)/(P₋−2P₀+P₊) НЕЗАВИСИМО по каждой оси N-D
  массива мощности, δ∈[−½,½], край оси → δ=0 без заворота; `axis_value_at` — дробный
  индекс → физ. значение оси; вход не мутируется). Демо `demo/ex5_peak_refine/`: 3 CW-цели
  на ДРОБНЫХ (kx,ky,f) (steering `ArrayGrid` — непрерывный фазовый вектор) → AmToCube
  32×32×256 → топ-3 NMS → парабола. **Метрики: ошибка argmax 0.29 бина → парабола 0.012
  (~×25), на clean И SNR 0 дБ.** PNG: cuts_* (срезы с параболой) + map_* (угловая карта
  + зумы 7×7). Тесты: test_peak_refine 7 ok (в т.ч. гаусс до 1e-9, интеграция с трактом),
  test_ex5 4 ok; бэкенд агрегатор зелёный, demo агрегатор зелёный. НЕ закоммичено.
  - **✅ ex5-web — ЖИВОЙ прототип web-панели (требование Alex: «всё двигается, не картинки»)**:
    `demo/ex5_peak_refine/web.py` → `demo/graphics/ex5_peak_refine/web/index.html` —
    САМОДОСТАТОЧНЫЙ HTML+JS (vanilla canvas, 850 КБ, ноль внешних URL — plotly дома нет,
    офлайн/Debian ок). Движение: реюз ex4 (`_MODEL_BUILDERS` core.motion + `_random_initial_state`
    + `Kinematics`→дробные бины; дальность→несущая, сближение = дрейф частотного бина).
    60 тактов × 2 SNR (clean/0 дБ), каждый такт: сцена→AmToCube 32×32×128→NMS→refine_peak.
    Панель: угловая карта+трейлы (argmax-«лесенка» серым vs парабола гладкая оранжевым),
    зум 7×7, срезы 3 осей с параболой, график ошибки по тактам, таблица; контролы
    play/pause/скорость/слайдер/SNR/цель/трейлы. **Метрики: argmax 0.394 бина → парабола
    0.013 (~×30), все 3 цели сопоставлены на всех 60 тактах.** Рендер проверен headless-Edge
    (скриншот — всё рисуется). Ex5WebTests 3 ok, demo агрегатор зелёный. НЕ закоммичено.

- ✅ **demo ex4 — летящие цель/помехи + барьер + анимация** (2026-07-18) — спека
  `specs/demo_ex4_flight_2026-07-18.md` (финал) + `tasks/TASK_demo_ex4_p1.md` (карта реюза).
  Sonnet упал 5-й раз — **Кодо написала сама**, реюз тотальный (motion 5 моделей + случайный
  выбор/старт по образцу demo_body_motion, эхо/гребёнка/null ex3, трекер core, тёмный стиль
  #0d1117, GIF FuncAnimation). Сущности: цель (WeavingManeuver) + носитель гребёнки
  (ConstantAccel) + barrage-дрейф. Такт: объём → полоса→null→грубая карта (патент «решает по
  дешёвой карте»; полный VolumeTokenizer только на финальном такте — фикс 17 мин→3м10с) →
  чистка точек (полоса ±3, гейт −15 дБ) → трекер. **Метрики: цель 30/30, полоса 30/30,
  треков 10 (№10 цель ЛЕТИТ + копии гребёнки), 3м10с.** Фиксы ревью: coarse_burst_points
  `max_points` (NMS ±2 — один argmax терял цель рядом с сильной гребёнкой; ex2/ex3 совместимы),
  дедуп детекций, срезы с признаками §4.11 в кадре. Выходы: flight_trail.gif / flight_clean.gif
  (хвост K=8 / без, из одной истории) + last_frame/trajectory PNG; plotly HTML — код есть,
  дома plotly не установлен (SkipTest). Тесты test_ex4 6 ok (полный размер, 6 тактов),
  **demo 38 ok, бэкенд 0 fail**. НЕ закоммичено.

- ✅ **demo ex3 — правильные эхо + ВСЕ 6 помех + null угла** (2026-07-18) — спека
  `specs/demo_ex3_echo_jammers_2026-07-18.md` (§0 решения + §8 ревью R1-R7). Sonnet упал 4-й раз
  (idle timeout) — **Кодо написала сама**. (1) **S1**: `build_pulse_echo_volume` в core
  (waveform_to_cube, по образцу ЛЧМ-прецедента): эхо = задержанная копия зонда, огибающая АМ
  с фронта (|фронт|=1+m=1.5, roll-эквивалент — тесты). (2) Помехи по одной за прогон, JNR +20/+10.
  (3) **R1-конвейер**: скан → полоса (столбец всплесков на угле) → гейт → **rank-1 null** угла
  (математика §4.1 SubspaceNuller при известном угле; EVD/MVDR на M=4096 непрактичны — честная
  девиация в докстринге, в тестах 16×16 сверка с EVD: подавление >60 дБ, остатки совпадают).
  **Метрики: ВСЕ 8 прогонов 6/6 объектов**; полоса на ТОЧНЫХ углах у 4 непрерывных помех
  (barrage/smsp/cw/vfd)→null, impulsive_arc — без полосы (физично). **DRFM-гребёнка (вопрос Alex
  «а гребёнка где?»)**: core-DrfmRepeaterJammer ретранслирует ЛЧМ-чирп (полоса в АМ-тракте;
  гребёнка из него — после дечирпа, гл.3) ⇒ для ex3 `build_drfm_comb_volume`: 5 затухающих
  эхо-копий НАШЕГО зонда (реюз build_pulse_echo_volume) — на сцене цепочка точек с шагом 0.1·N,
  drfm false=31 = ложные цели гребёнки (их отсекает FM-m арбитр гл.5, вне ex3). Прогон 64 с,
  13 PNG (before/after). **Фикс ревью в ex2**: порог грубого скана = статистика МАКСИМУМА куба
  `T=N̂·ln(n/pfa)` (per-cell порог давал всплеск-мусор после null). demo 32 ok, бэкенд 0 fail.
  НЕ закоммичено (+ незакоммичены правки графики: scene_points в core/graphics, теплокарты 6/6).

- ✅ **demo ex2 — сигналы в апертуре 64×64×4096, детекция гл.4-бис** (2026-07-18) — спека
  `specs/demo_ex2_cube_2026-07-18.md` (§0 решения Alex + §7 ревью Кодо R1-R8) → Sonnet (упал на
  обрыве, Кодо дочитала/доделала) → ревью Кодо. `demo/ex2_am_square/`: 6 объектов (am+radio ×
  4/8/16 пер., steering-эхо комплексно), двухэтапный скан патента §4-бис.2а (грубо 32/32 дешёвый
  Exp-порог → ROI-слияние → тонко 16/8 ПОЛНЫЙ VolumeTokenizer), DataContext gen|disk + Observer
  канал "volume". **Фикс ревью Кодо: динамический гейт −20 дБ** (на clean боковики steering давали
  176 false → 0). **Метрики: clean 6/6 false 0 · +10 дБ 6/6 false 0, угол 0 бинов, контраст 36 дБ.**
  Прогон 27 с, 4 PNG (3D-сцена, врезки B/E с шумом, теплокарты clean/+10). test_ex2 5 ok,
  demo 26 ok, бэкенд 0 fail. НЕ закоммичено.

- ✅ **Угловая кластеризация детекций** (2026-07-18) — `core/models/anti_barrage/clustering.py`:
  `DetectionCluster` (VO) + `DetectionClusterer` (single-linkage union-find по близости угол+дальность,
  центроид = пик level_db). Один источник (лепесток/страддл занимает неск. ячеек) → 1 кластер, убирает
  дубли для трекинга. Проверено Кодо: 3 соседние+2 отдельные→3 кластера. test_clustering 6 ok, бэкенд 0 fail.

- 🔨 **demo-серия ex1 (стенд `demo/core/` + `Ex1AmLine`)** (2026-07-18) — demo теперь ведёт ЭТОТ
  чат (Alex переназначил). Ревью Кодо спеки `specs/demo_ex1_signal_2026-07-18.md` → **§0 решения
  Alex**: длительности 4/8/16 · ex1 переписать под `DemoRunner`(Template Method) · f_c=250 оставить
  с честной пометкой алиасинга · фикс бага `fig_three_variants` (Вариант2=Вариант3). Делегировано
  Sonnet (стенд `demo/core/`: DemoRunner/DemoContext/DemoWriter/DemoReport + `Ex1AmLine` + тесты +
  `run_all.py`; SceneBank/placement НЕ трогаем — с ex2).
  - **✅ ПРИНЯТО (ревью Кодо, сама прогнала):** стенд+пример по §0, реюз фабрики честный,
    **demo 4 ok · весь бэкенд 0 fail · run_all=16 PNG**. Sonnet упал на обрыве (idle timeout) —
    Кодо дочитала состояние и доделала: `all_demo_test.py` (не создан) + `sys.path`-хук в 4 точки
    входа (конвенция репо, Sonnet упустил) + чистка 8 старых PNG прошлого скрипта.
  - **⚠️ Находка ревью (ошибка Кодо в §0 признана):** «баг Вариант2=Вариант3» — НЕ баг: для АМ
    `|a(t)|=1+m·cos` при m≤1 ⇒ **огибающая ≡ магнитуда** тождественно (smoke: разница 1.4e-7).
    Alex: «вывести все 3, осмотрю» — оставлено как есть, PNG `demo/graphics/ex1_am_line/
    {am,radio}_three_variants.png`. Различие Вар2/3 появилось бы только при перемодуляции m>1.
  - **НЕ закоммичено** (за Alex). `test_integration.py`/`all_test.py` в статусе — чужой коммит
    (`3cf1d34`, параллельный чат core-базы), не трогала.
  - **✅ Ветка ex1 переделана по уточнениям Alex (2026-07-18):** длительность = 4/8/16 периодов
    НЕСУЩЕЙ f_m (буквально §2, «не видно — так нужно»); все 3 длительности на ОДНОЙ оси 4096
    (t0=300/1600/2900); 12 PNG парами clean+noise (2 типа × 3 f_m × 2).
  - **✅ ex1-denoise (слепая детекция в белом шуме)** — спека `specs/ex1_denoise_2026-07-18.md`
    (§0: детектор знает только «шум белый» + тип из семейства; несущая ГДЕ УГОДНО в ±250 МГц).
    Sonnet 2× упал на обрыве, **Кодо написала сама**: `demo/ex1_am_line/denoise.py` —
    `SpectralGateFilter` (C: порог T=−N̂·ln(pfa) над Exp-полом спектра) + `WienerFilter` (D) +
    `PulseDetector` (реюз `OsCfarDetector` core на 1D, gap_tol=32, гейт −20 дБ sidelobe-blanking
    против звона Гиббса — диагностика: ложные были −23…−53 дБ). 6 PNG `ex1_denoise/`
    [вход|C|D]×SNR (правка Alex). Метрики: C 71/108 found, D 74/108 (ниже +3 дБ слепой предел —
    физика). Тесты `test_denoise.py` 5 ok, demo 9 ok, **бэкенд 0 fail**. НЕ закоммичено.
  - **✅ ex1-stft (СТФТ-детекция, ТЗ Alex)** — `demo/ex1_am_line/stft_detect.py`: окно 16×Хэмминг
    + 16 нулей (FFT-32), шаг 8 (оверлап 50%) → спектрограмма → Exp-порог на ячейку → сигнальные
    кадры → сегменты (кадр→отсчёт через hop). **Лучший из трёх: found 83/108, false 0**
    (vs C 71/23, D 74/4) — выигрыш бина ≈ +12 дБ: до +3 дБ все 3/3, на 0 дБ 2/3. f̂ по argmax
    среднего спектра (разрешение fs/32=15.6 МГц). 6 PNG `ex1_stft/` [вход | спектрограмма+рамки].
    `test_stft.py` 6 ok, demo 15 ok, бэкенд 0 fail. НЕ закоммичено.

- ✅ **ROI-гейт детекции** (2026-07-18) — `core/models/targeting/roi_gate.py`: `RoiGate` фильтрует
  `list[Detection]` по зоне интереса вокруг целеуказания (`BeamCommand.center` ± angle_half по углу,
  target_r ± range_half по дальности; union по beams) — убирает ложные детекции вне ROI (§8 гейтит
  детекцию). test_roi_gate 7 ok, весь бэкенд 0 fail. Замыкает целеуказание→CFAR.


- ✅ **Robust MVDR-nuller** (2026-07-18) — `core/models/anti_barrage/mvdr.py`: `RobustMvdrNuller`
  (Capon `w=R⁻¹a/(aᴴR⁻¹a)` через `np.linalg.solve`, diagonal loading в R). Замыкает находку phase2:
  здесь loading **реально критичен** (обращение R⁻¹), в отличие от subspace-проекции (no-op).
  Проверено Кодо: distortionless `|wᴴa|=1.0`, loading стабилизирует `‖w‖` (1.68→0.18, ~9×) на
  вырожденной R (K<M), заград подавлен. test_mvdr 4 ok, весь бэкенд 36 наборов 0 fail.


- ✅ **anti-barrage phase2 (pipeline + diagonal loading)** (2026-07-18) — `AntiBarragePipeline`
  (Facade: nuller→Fft3DModel→cfar единым `process`) + diagonal loading в `SubspaceNuller`.
  **⚠️ Находка Кодо (математика):** diagonal loading НЕ влияет на подавление (`apply` инвариантен —
  проекция на собственные ВЕКТОРЫ, `R+λI` их не меняет), влияет ТОЛЬКО на `report.lambda_ratio`/
  детектор `is_barrage`. Робастное подавление при малой K = MVDR (обращение R⁻¹) — отдельная задача.
  Тест документирует находку. **Весь бэкенд 35 наборов, 215 ok, 0 fail.**
  → [`specs/anti_barrage_phase2_2026-07-18.md`](../specs/anti_barrage_phase2_2026-07-18.md).
  - 🎉 **ВСЯ БАЗА (core) ПО ЛОГИКЕ ЗАКРЫТА:** нормировка признаков · арбитр гл.5 (геом.+код) ·
    трекинг · OS-CFAR точная Pfa · целеуказание гл.8 · калибровка §4.12 · anti-barrage phase2.
    Осталось: активный FM-m опрос (с приёмника, вне прототипа), L3/CNN (torch — не дома).


- ✅ **Калибровка/валидация триажа на датасете (§4.12)** (2026-07-18) — `core/models/tokenizer/
  calibration.py`: `TriageCalibrator` (build_dataset классов source/noise/smeared через angular_fft
  + steering, разные апертуры/SNR/углы) + `validate` (confusion/accuracy/pfa_noise). Проверено Кодо
  независимо (seed 42): **source/noise accuracy 1.0, pfa_noise 0.0** на 16×16/64×64/128×64; smeared
  0.87→1.0. Якоря §4.11 НЕ тронуты — валидны на датасете. **Инвариантность к M доказана ⇒ якорь
  SOURCE для больших M НЕ нужен** (метка не деградирует, нюанс был только в score). CalibrationTests
  6 ok, бэкенд 0 fail. НЕ закоммичено.


- 🎯 **Целеуказание пучка FM-m (гл.8) — замыкает когнитивную петлю** (2026-07-18) — новый пакет
  `core/models/targeting/`: `BeamCommand` (VO: target_r/center/beam_angles), `Targeting(ABC)`,
  `BeamTargeting` (пучок лучей в конус неопределённости вокруг цели, `cone_half`/`max_beams`, только
  `decision=="target"`), `CognitiveCycle` (Facade §8.3: tokenize→arbitrate→target). test_targeting
  9 ok (BeamTargeting 7 + CognitiveCycle 2), весь бэкенд 0 fail. Sonnet упал 2× на обрыве — Кодо
  дочитала код + написала тесты сама. 🎉 **Пайплайн базы замкнут: тракт i×j → токенизатор →
  арбитр (геом.+код) → трекинг → целеуказание.** Дальше активный опрос — с приёмника (вне прототипа).


- ✅ **OS-CFAR точная Pfa (Rohling)** (2026-07-18) — исправлена честная девиация в
  `core/models/tokenizer/cfar.py`: α по CA-приближению → **точная OS-CFAR** (`_os_pfa` формула
  Rohling `∏(n-i)/(n-i+α)` + `_alpha_os` bisection + k-я порядковая статистика через `np.partition`,
  кеш α по (n_eff,k)). Проверено Кодо: обратимость точная (α↔Pfa), эмпирич. Pfa 0.0107≈0.01,
  α_OS=7.42 vs α_CA=8.64 (CA игнорировал k). OsCfarPfaTests 4 ok, весь бэкенд 0 fail. НЕ закоммичено.


- 🎯 **Трекинг детекций между тактами (§4-бис.4 «летит»)** (2026-07-18) — новый пакет
  `core/models/tracking/`: `Track` (VO: history/vel_r/vel_angle/is_moving), `Tracker(ABC)`,
  `NearestNeighborTracker` (жадная NN-ассоциация по гейту `Δkx²+Δky²+(w_r·Δr)²`, MNK-скорость,
  рождение/смерть по `max_missed`). Связывает решения арбитра (`TargetDecision=="target"`) между
  тактами → траектория + «летит» (движение из треков, НЕ из куба, §4-бис.4/§5.7). test_tracking
  7 ok, весь бэкенд 0 fail (Sonnet+ревью Кодо). Проверено: движ. цель→1 трек vel_r=-10/is_moving,
  две цели не путаются, пропажа→смерть. ⚠️ `w_r` — вес масштаба дальности (бины на порядок больше
  угла); дефолты юзабельны. Задел: предсказание по vel (без Калмана — §4-бис.4 не требует).


- 🎯 **Физический арбитр гл.5 (передний край τ≥0)** (2026-07-18) — новый слой пайплайна после
  токенизатора: `core/models/tokenizer/arbiter.py`. `EdgeArbiter` (Вариант 1, геометрия): `comb`→
  передний край=истинная цель, `barrage`→jammer, одиночка→target. `CodeArbiter` (Вариант 2, FM-m
  код) — задел (LSP). test_arbiter 7 ok, весь бэкенд 0 fail (Sonnet+ревью Кодо).
  → [`specs/arbiter_edge_ch5_2026-07-18.md`](../specs/arbiter_edge_ch5_2026-07-18.md).
  - **✅ Вариант 2 (`CodeArbiter`, свежесть FM-m кода) СДЕЛАН (2026-07-18):** `fm_correlate` (numpy,
    гл.6 §6.2) + `CodeArbiter` (свежий код→target, чужой/шум→false) + `CombinedArbiter` (§5.4:
    геометрия И код). **Чистит осколки:** цель 34.9 дБ→target, осколок 12 дБ→false (проверено Кодо).
    test_arbiter 15 ok (Edge6+Code6+Combined3), весь бэкенд 0 fail. Тонкость: чужой код = другой
    полином, НЕ другой seed (seed = сдвиг фазы того же кода). 🎉 **Арбитр гл.5 замкнут (Вар.1+Вар.2).**
  - **Следующее (на будущее):** интеграция арбитра в `SceneServer` + реальный FM-m опрос (гл.8) +
    трекинг между тактами (§4-бис.4 «летит»).


- 🔧 **Рефактор апертуры: 16×16 → i×j / 2ⁿ / zero-pad** (2026-07-17) — новая концепция: апертура
  не квадрат, а i×j (nx≠ny), каждая ось 2ⁿ, недобор → zero-pad; угловой FFT паддит до 2ⁿ,
  `sinθ=k/(N_pad/2)` по осям независимо. Скилл-оркестратор `/aperture-refactor`.
  → аудит [`specs/aperture_ixj_2n_refactor_2026-07-17.md`](../specs/aperture_ixj_2n_refactor_2026-07-17.md).
  - **✅ E2 (код тракта):** `angular_fft`/`fft3d`/`waveform_to_cube` паддят до 2ⁿ (Sonnet+ревью Кодо).
    Проверено сам: 6×15→8×16 (LFM+AM), 16×16 no-op. `ArrayConfig.padded_shape()` теперь вызывается.
  - **✅ E1 (общий файл):** `Doc/Patent/00_КОНЦЕПЦИЯ_ixj_2n.md` — что переделать построчно (17 глав).
  - **✅ E1-шапки** (17 патент-глав, Sonnet+ревью). Минор: YAML-фронтматтер в 1 заявке перед пометкой.
  - **✅ E3-тесты** концепции: `tests/test_aperture_ixj.py` (7 тестов, проверил сам 7 ok/0 fail —
    padded_shape/angular_fft паддит/zero-pad=нули/боресайт→центр). Регресс зелёный.
  - **✅ угл. шкала** `sinθ=k/(N_pad/2)` по осям — в SPEC §1.1.
  - **✅ E4:** текст 9 спек/тасков 16×16→i×j + `M=256→M=nx·ny` + подсекция угл.шкалы в TASK_p5
    (Sonnet+ревью). Формула `sinθ=k/(N_pad/2)`, `Δsinθ=2/N_pad` (16×16→k/8, сходится с патентом).
  - **⚠️ Исправлена ошибка Кодо:** формула была `sinθ=k/N_pad` (фактор 2 неверно) → `k/(N_pad/2)`,
    поправлено в 24 файлах (спеки/таски/патенты/скилл/докстринг). Визуал `graphics/aperture_ixj/`.
  - **✅ Финальный прогон:** test_aperture_ixj/waveform_to_cube/tokenizer/runtime — все зелёные.
  - **✅ E5 (демо):** `demo_cfar/nuller/tokenizer` получили CLI `--nx/--ny` (дефолт 16). Проверено:
    все 3 демо зелёные на дефолте 16×16 И на неквадрате 6×15→pad 8×16 (Sonnet упал на обрыве — Кодо дочитала/доделала).
  - **✅ E6 (докстринги):** «16×16»→«nx×ny» в core (square_view/panel/cfar/waveform_to_cube/project_config).
  - **🎉 РЕФАКТОР АПЕРТУРЫ i×j/2ⁿ/zero-pad ПОЛНОСТЬЮ ЗАКРЫТ (E1–E6).** Ядро закоммичено (`9817a27`,
    запушено). Хвост E5/E6 — НЕ закоммичен (за Alex).
  - **Ключевое:** WMMA-плитки 16×16 (железо) и дальностный zero-pad НЕ трогаем.


- 🎯 **Объёмный токенизатор (OS-CFAR 3D + 6 признаков патента)** (2026-07-16) — спроектирован
  + глубокое ревью Кодо (сверка с патентом гл.4 + 4-бис). Найдено/исправлено 6 расхождений
  плана (E1–E6: токен на срез не на куб · 2 прохода · LobeRatio-формула · OS-CFAR=4-бис ·
  RuleBased вместо MLP). 6 признаков (PR/Hoyer/MainFrac/LobeRatio/MaxMean/Energy) в коде
  **НЕ реализованы** — текущий `SquareView.tokenize` = заглушка (−10 дБ + argmax).
  → [`TASK_tokenizer_p1.md`](TASK_tokenizer_p1.md).
  - **Ход:** Sonnet реализовал `core/models/tokenizer/` (features/cfar/triage/tokenizer/tokens) →
    глубокое ревью Кодо 2026-07-16. **169 ok/0 fail/9 skip**, ruff чистый. Признаки сверены
    Кодо НЕЗАВИСИМО против §4.11: цель PR=3.5/MainFrac=1.0/LobeRatio=0.0001 и шум PR=120/Hoyer=0.34
    ≈ патент (не подгонка). LobeRatio (E4) и P=mag² (E7) реализованы верно. 2 прохода (target/comb/
    barrage + автокорр) ок. Девиации честно задокументированы (α по CA-приближению; триаж —
    параллельный ABC `SliceTriage`, не наследует `CubeClassifier`, чтобы не ломать LSP).
  - **✅ Заград-хвост ЗАКРЫТ (2026-07-17, диагностика Кодо → Sonnet-фикс → ревью Кодо):** прошлая
    находка «smeared слабый» переопределена. Диагностика на РЕАЛЬНОМ `BarrageRfJammer`: проход 1
    даёт `source` (заград собран по УГЛУ — верно, §4-бис.4), smeared-порог калибровать не нужно.
    **Настоящий баг был в проходе 2:** `assemble_range` разваливал направленный заград на ложные
    `target`/`comb` (угловой джиттер пика ломал строгое `span==n`). Фикс: barrage по `fill≥0.7`
    (не строгая непрерывность). boresight → `barrage` (было `target`), RangeAssemblyTests 7 ok.
    Осколки джиттера (~144 target) → арбитр гл.5 (FM-m код), вне прохода 2 (§4.9/§4.12).
    → [`specs/tokenizer_barrage_pass2_2026-07-17.md`](../specs/tokenizer_barrage_pass2_2026-07-17.md).
  - **✅ S5 ЗАКРЫТ (2026-07-17, Sonnet-код → ревью Кодо):** интеграция в `SceneServer`/панель.
    `SceneServer` DI-инъекция `VolumeTokenizer(window_l=1)`, в `step()` публикует 4-й канал
    `"tokens"` (`_tokens_payload` — чистые примитивы N2, roundtrip через `codec` в тесте).
    `PanelModel.ingest_tokens` + `SignalBlock.slice_tokens`/`verdict` (аддитивно, `SquareView`/
    `SquareToken` контрольный вид НЕ тронут). Ревью Кодо независимо: `index_of_angle` инвертирует
    точно, панель несёт реальные токены детектора + вердикт прохода 2. **test_runtime 26 ok/0 fail,
    test_tokenizer 13 ok/0 fail, test_graphics/waveform_to_cube зелёные** (подтверждено Кодо).
    ⚠️ Среда дома: нет scipy (`test_generators`/`all_test` падают на импорте — не регресс) и ruff.
    ⚠️ Заметка на будущее: `tokenize` на каждом такте по всему кубу 16×16×1024 — для дома ок,
    позже ROI-ограничение/шаг. **🎉 Токенизатор P1 (S1–S5) ПОЛНОСТЬЮ ЗАКРЫТ.**


- 🔨 **P6 · Движение тела в 3D-кубе + помехи + сокет-панель** (2026-07-15) — спека
  **финализирована** (сверена с патентом гл.3+4-бис; ревью F1–F10).
  → спека [`specs/body_motion_3d_2026-07-15/SPEC.md`](../specs/body_motion_3d_2026-07-15/SPEC.md)
  + `scheme.svg`; ревью [`specs/body_motion_3d_review_2026-07-15.md`](../specs/body_motion_3d_review_2026-07-15.md).
  - **Ключевое:** ЛЧМ = 2 раздельных FFT (дальностный глобальный + угловой 16×16, **без окна**);
    АМ = локальный 3D-FFT по окну `16×16×D` (шаг 8/16/32/64). Общее — куб-примитив + токенизатор.
    `DataContext` = базовый класс обмена (расширяем; шина `Blackboard`/Observer отдельно, SRP).
    Транспорт ZMQ+MessagePack. Движение: cv+Markov, без рывков. Проектируем сразу под C++.
  - **Таски (Sonnet-код → глубокое ревью Кодо):**
    P1 [`TASK_body_motion_p1.md`](TASK_body_motion_p1.md) — фундамент (ProjectConfig+шина+motion). ✅ РЕАЛИЗОВАН (код+тесты зелёные, проверено Кодо 2026-07-17).
    P2 [`TASK_body_motion_p2.md`](TASK_body_motion_p2.md) — splat цели → входы веток nx×ny×N. ✅ РЕАЛИЗОВАН (`generators/volume.py`, test_body_motion_volume зелёный).
    P3 [`TASK_body_motion_p3.md`](TASK_body_motion_p3.md) — помехи (заград+гребёнка). ✅ ПРИНЯТО (ревью Кодо 2026-07-15).
    P4 [`TASK_body_motion_p4.md`](TASK_body_motion_p4.md) — несколько целей (Composite). ✅ ПРИНЯТО (ревью Кодо 2026-07-15).
    P5 [`TASK_body_motion_p5.md`](TASK_body_motion_p5.md) — WaveformToCube (ЛЧМ 2FFT / АМ 3D-FFT). ✅ РЕАЛИЗОВАН (`waveforms/waveform_to_cube.py`, test_waveform_to_cube зелёный).
    P6 [`TASK_body_motion_p6.md`](TASK_body_motion_p6.md) — сокет-панель (ZMQ Observer + Dear PyGui). ✅ ПРИНЯТО (ревью Кодо 2026-07-16).
    🎉 **body_motion P1–P6 ПОЛНОСТЬЮ ЗАКРЫТ.**
  - **🔬 Глубокий анализ тасков (2026-07-15):** [`specs/body_motion_3d_tasks_review_2026-07-15.md`](../specs/body_motion_3d_tasks_review_2026-07-15.md)
    — 9 находок, все внесены. Ключевое: **A9** — P2 реюзит готовые Python-генераторы `waveforms/`
    (`WaveformFactory`+`render`+`SignalField`+`NumpyBackend`, вчера, 48+ тестов), не изобретает splat;
    numpy не применяет `tau_s` → дальность окном; дечирпа нет → добавить в P5. Также A1 (реюз `Scene`),
    A4 (ЛЧМ=`LfmWaveform`/АМ=`AmWaveform`), A5 (YAML через `run_workspace`).
  - **Процесс:** ревью тасков → skill с Sonnet-агентами по таскам → глубокое ревью Кодо. Старт — S1.

- ✅ **Настройка конфигурации проекта** (2026-06-22) — перенос правил/хуков/MCP/стиля из
  DSP-GPU + rag-mentor, починен баг `Core/` → `core/`. **Готово.**

- ✅ **Рефакторинг графики phase1** (2026-07-07) — реализовано Sonnet + ревью Кодо. Все 7 шагов
  зелёные (mypy/ruff/тесты/main/demo). **Готово.** → [`TASK_graphics_refactor_phase1.md`](TASK_graphics_refactor_phase1.md)

- ✅ **anti-barrage · SubspaceNuller phase1** (2026-07-07) — реализовано Sonnet + ревью Кодо.
  Угловое подавление barrage −20.3 дБ, цель выживает. **Готово.**
  → [`TASK_subspace_nuller_phase1.md`](TASK_subspace_nuller_phase1.md)

- ✅ **anti-barrage · CA-CFAR phase1** (2026-07-07) — детектор по дальности + local-max гейт.
  Ревью Кодо вскрыло self-masking, фикс `n_guard=4`+localmax. Цель детектируется. **Готово.**
  → [`TASK_cfar_phase1.md`](TASK_cfar_phase1.md)

- ✅ **SNR-эстиматор phase1** (2026-07-07) — порт из GPUWorkLib: `core/snr/` (спектр CA-CFAR +
  статистика), 3 графика, torch-free. Инструкция `Doc/snr_usage.md`. **Готово.**
  → [`TASK_snr_phase1.md`](TASK_snr_phase1.md)

- 🔨 **Генераторы сигналов АМ/ЛЧМ/ФМн + помехи** (2026-07-14) — спека согласована + ревью закрыто.
  Схема: Кодо таски → ревью → Sonnet код → Кодо проверка.
  → спека [`specs/signal_generators_2026-07-13.md`](../specs/signal_generators_2026-07-13.md)
  → ревью [`specs/signal_generators_review_2026-07-14.md`](../specs/signal_generators_review_2026-07-14.md)
  - **Ход реализации (Sonnet код → Кодо ревью):**
    P0 [`TASK...p0.md`](TASK_signal_generators_p0.md) — ✅ ПРИНЯТО (SignalField+конфиг+окно+YAML; 11 тестов).
    P1 [`TASK...p1.md`](TASK_signal_generators_p1.md) — ✅ ПРИНЯТО (NumpyBackend+CW/ЛЧМ/АМ; 48 тестов, SNR ок).
    P2 [`TASK...p2.md`](TASK_signal_generators_p2.md) — ✅ ПРИНЯТО (ревью Кодо 2026-07-14). GPU на
    RX 9070: python3.13 → **26 ok/0 fail**; .venv(cp312) → 20 ok + 6 skip (GPU скип чисто, ABI не падает).
    Эскалация G11 закрыта (порог 1e-3, ЛЧМ-канон=`getX` центр., `norm=1.0`). CW/ЛЧМ N=4096 raw=5.11e-4
    (<1e-3); Кодо добавил тест **модуля на baseline N=8192 = 1.19e-7** (закрыл дыру находки — фаза
    дрейфует float32, но модуль точен, и в дечирпе rx·conj(ref) сокращается). `HipBackend` DRY (реюз
    `render_pipeline`), `.so` в gitignore, вендоринг с атрибуцией.
    P4 [`TASK...p4.md`](TASK_signal_generators_p4.md) — ✅ ПРИНЯТО (ФМн-код+коррелятор+ЧМ). Автокорр
    thumbtack (пик 8191); LFSR==GPU `generate_msequence` побитово; коррелятор-интероп на RX 9070 пик
    на d=250; .venv 31 ok/8skip, python3.13 39 ok/0. H1/H2/H3 (правки Alex) отработали.
  - **✅ Фундамент P0–P2 + ФМн P4 готовы**: генерация CW/ЛЧМ/АМ/ФМн + ЧМ на n×n, 2 бэкенда, коррелятор-реюз,
    конфиг+YAML, `SignalField`. P3(окно/шум) фолдится в P1.
    P5 [`TASK...p5.md`](TASK_signal_generators_p5.md) — ✅ ПРИНЯТО (6 помех: BarrageRF/SMSP/DRFM/INT_CW/
    IMP_ARC/VFD). python3.13 47 ok/0; barrage rank-1, IMP_ARC kurtosis=260, DRFM 4 пика. J1-J7 (Alex) ок.
  - **✅✅ ГЕНЕРАЦИЯ (§0) ЗАВЕРШЕНА**: все типы (CW/ЛЧМ/АМ/ФМн/ЧМ) + шум + помехи (патент+промышленные)
    на n×n, 2 бэкенда (numpy+GPU), коррелятор-реюз, окно/размещение, YAML. Осталось: **P6** (такты/движение,
    ⏳ ресёрч Q7) + **P7** (порт в C++). Оба Alex пометил «позже».
  - 🐞 **Хвост:** `.gitignore` баг (инлайн-коммент) починен Кодо; нужен `git rm -r --cached graphics/`
    (8 PNG p0-p2 затрекались) — за Alex, перед следующим коммитом.
  - 🖼️ **Графики-подтверждения:** корневой `graphics/signal_generators/<phase>/` (подкаталог на фазу,
    НЕ в куче), пишем `FigureWriter` (сам mkdir), `graphics/` в `.gitignore`. Каждая фаза даёт визуал.
  - **Роадмап (детализируем после P0–P2):**
    P3 — TimeWindow full/partial/short + AdditiveNoise (SNR-в-дБ, R5). 🖼️ `p3_window_noise/`.
    P4 — ФМн-код 2ⁿ (LFSR, наш) + коррелятор `FMCorrelatorROCm` (реюз, §6.2) + ЧМ + автокорр. 🖼️ `p4_phase_code_fm/`.
    P5 — помехи: патент (SMSP/DRFM-repeater/BarrageRF) + промышленные (INT_CW/IMP_ARC/HAR_VFD…, §3.1). 🖼️ `p5_jammers/`.
    P6 — `TactSequence`: движение цели + помехи во времени (+ ⏳ ресёрч моделей движения, Q7). 🖼️ `p6_tacts/`.
    P7 — порт прототипа в DSP-GPU (C++/HIP): АМ/ФМн/ЧМ + движение (§2.2).
    P8 — `demo_generators.py` + сводные графики + полный прогон тестов. 🖼️ `p8_demo/`.
  - **Блокеры ревью:** R1 (загрузка .so → `core/gpu_libs/`, копируем) — решено; R2 (реальный GPU-smoke) — шаг 0 в P2.

## Следующее

- 🎵 **Доплер (§5)** — тяжёлый torch-этап: +ось импульсов (RangeConfig+n_pulses), скорость цели,
  БПФ-4D, признак «летящий»; разделяет даже mainlobe-помеху. Далее 3D-CNN (§8).
- 🔧 (фон) phase2 фильтров: впайка nuller/CFAR в pipeline, torch/GPU-бэкенд, угловая кластеризация,
  diagonal loading для nuller, гейт детекции по ROI (убрать остаточные ложные в barrage).
