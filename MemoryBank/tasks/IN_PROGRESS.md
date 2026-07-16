# 🔄 IN_PROGRESS

> Короткий указатель на активную задачу (1–5 строк). Детали — в `TASK_<topic>_<phase>.md`.
>
> 🤝 **Передача дел (2026-07-15, продолжать на Linux):** `sessions/2026-07-15_HANDOFF_P3P4_to_P6.md`.
> P1–P5 ✅, P3+P4 приняты в этой сессии. Остался **P6** (сокет-панель). Не закоммичено (Alex пушит).
> ⚠️ Перед git: `rm -f .git/index.lock _sync_test.txt`.

## Сейчас в работе

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
  - **⚠️ Находка ревью:** триаж класса **smeared/заград слабый** — на игрушечном заграде скор 0.50
    (честно «не уверен»), т.к. модель заграда ≠ патентной. Калибровать пороги на РЕАЛЬНЫХ
    заград-кубах (`generators/jammers` BarrageRF) — совпадает с §4.12 (пороги калибруются на датасете).
  - **⏳ Осталось (S5):** интеграция в `SceneServer`/панель (заменить заглушку `SquareView.tokenize`
    на реальные `SliceToken`) — Sonnet-агент оборвался на API-ошибке до S5. Визуал `feat_scene.png` ✅ (Кодо).


- 🔨 **P6 · Движение тела в 3D-кубе + помехи + сокет-панель** (2026-07-15) — спека
  **финализирована** (сверена с патентом гл.3+4-бис; ревью F1–F10).
  → спека [`specs/body_motion_3d_2026-07-15/SPEC.md`](../specs/body_motion_3d_2026-07-15/SPEC.md)
  + `scheme.svg`; ревью [`specs/body_motion_3d_review_2026-07-15.md`](../specs/body_motion_3d_review_2026-07-15.md).
  - **Ключевое:** ЛЧМ = 2 раздельных FFT (дальностный глобальный + угловой 16×16, **без окна**);
    АМ = локальный 3D-FFT по окну `16×16×D` (шаг 8/16/32/64). Общее — куб-примитив + токенизатор.
    `DataContext` = базовый класс обмена (расширяем; шина `Blackboard`/Observer отдельно, SRP).
    Транспорт ZMQ+MessagePack. Движение: cv+Markov, без рывков. Проектируем сразу под C++.
  - **Таски (Sonnet-код → глубокое ревью Кодо):**
    P1 [`TASK_body_motion_p1.md`](TASK_body_motion_p1.md) — фундамент (ProjectConfig+шина+motion). ⏳
    P2 [`TASK_body_motion_p2.md`](TASK_body_motion_p2.md) — splat цели → входы веток 16×16×N. ⏳
    P3 [`TASK_body_motion_p3.md`](TASK_body_motion_p3.md) — помехи (заград+гребёнка). ✅ ПРИНЯТО (ревью Кодо 2026-07-15).
    P4 [`TASK_body_motion_p4.md`](TASK_body_motion_p4.md) — несколько целей (Composite). ✅ ПРИНЯТО (ревью Кодо 2026-07-15).
    P5 [`TASK_body_motion_p5.md`](TASK_body_motion_p5.md) — WaveformToCube (ЛЧМ 2FFT / АМ 3D-FFT). ⏳
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
