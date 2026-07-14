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

- 📝 **Генераторы сигналов АМ/ЛЧМ/ФМн + помехи (план)** (2026-07-13) — черновик спеки на правку Alex.
  Слой сырого времени + мост в куб, 2 бэкенда (NumPy/HIP), train/deploy split (cp312/cp313).
  → [`specs/signal_generators_2026-07-13.md`](../specs/signal_generators_2026-07-13.md) · **6 открытых вопросов в §11**

## Следующее

- 🎵 **Доплер (§5)** — тяжёлый torch-этап: +ось импульсов (RangeConfig+n_pulses), скорость цели,
  БПФ-4D, признак «летящий»; разделяет даже mainlobe-помеху. Далее 3D-CNN (§8).
- 🔧 (фон) phase2 фильтров: впайка nuller/CFAR в pipeline, torch/GPU-бэкенд, угловая кластеризация,
  diagonal loading для nuller, гейт детекции по ROI (убрать остаточные ложные в barrage).
