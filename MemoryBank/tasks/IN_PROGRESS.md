# 🔄 IN_PROGRESS

> Короткий указатель на активную задачу (1–5 строк). Детали — в `TASK_<topic>_<phase>.md`.

## Сейчас в работе

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
    конфиг+YAML, `SignalField`. P3(окно/шум) фолдится в P1. Дальше: **P5** (помехи: патент + промышленные).
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
