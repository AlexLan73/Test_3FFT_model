# 🔄 IN_PROGRESS

> Короткий указатель на активную задачу (1–5 строк). Детали — в `TASK_<topic>_<phase>.md`.

## Сейчас в работе

- ✅ **Настройка конфигурации проекта** (2026-06-22) — перенос правил/хуков/MCP/стиля из
  DSP-GPU + rag-mentor, починен баг `Core/` → `core/`. **Готово.**

- ✅ **Рефакторинг графики phase1** (2026-07-07) — реализовано Sonnet + ревью Кодо. Все 7 шагов
  зелёные (mypy/ruff/тесты/main/demo). **Готово.** → [`TASK_graphics_refactor_phase1.md`](TASK_graphics_refactor_phase1.md)

## Следующее

- 🎯 **anti-barrage · SubspaceNuller** (шаг 1 плана внедрения) — детерминир. угловое подавление
  (oblique projection) на текущем кубе, GPU-ready на torch. → план: memory `anti-barrage-plan`,
  спека [`specs/anti_barrage_detection_2026-06-23.md`](../specs/anti_barrage_detection_2026-06-23.md)
