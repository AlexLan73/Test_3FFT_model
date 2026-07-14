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
  - **Детальные таски готовы (на ревью Alex):**
    P0 [`TASK_signal_generators_p0.md`](TASK_signal_generators_p0.md) — SignalField+конфиг+окно+YAML.
    P1 [`TASK_signal_generators_p1.md`](TASK_signal_generators_p1.md) — NumpyBackend+CW/ЛЧМ/АМ.
    P2 [`TASK_signal_generators_p2.md`](TASK_signal_generators_p2.md) — gpu_libs+GPU-smoke+HipBackend+сверка.
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
