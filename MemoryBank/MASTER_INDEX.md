# 🗂️ MASTER_INDEX — radar3d (Test_3FFT_model)

> Читать **первым** в начале каждой сессии. Карта состояния проекта.

## 📊 Статус

- **Проект**: radar3d — пространственно-временна́я 3D-БПФ обработка + каскад-классификатор.
- **Стадия**: каркас (слои + паттерны), RuleBased-классификатор работает без torch.
- **Конфиг**: гармонизирован с DSP-GPU и rag-mentor (правила, хуки, MCP, стиль).

## 🧭 Навигация

| Что | Где |
|-----|-----|
| Правила Кодо | `.claude/rules/*.md` (6 файлов) |
| Архитектура / запуск | `CLAUDE.md`, `README.md` |
| Документация | `Doc/` — [C4](../Doc/architecture/README.md), [папки](../Doc/folders.md), [классы](../Doc/classes.md) |
| Активные задачи | `MemoryBank/tasks/IN_PROGRESS.md` |
| Спеки / ревью / исследования | `MemoryBank/specs/` |
| Журнал сессий | `MemoryBank/sessions/YYYY-MM-DD.md` |
| Changelog | `MemoryBank/changelog/YYYY-MM.md` |

## 🗂️ Код

- `core/config/` — конфиги (Value Object)
- `core/generators/` — сигналы + помехи (Factory/Builder/Composite)
- `core/models/` — 3FFT (Template Method) + `classification/` (Strategy/LSP)
- `core/graphics/` — визуализаторы (Strategy)
- `core/data_context/` — load/save (Facade/Repository)
- `core/controller.py` — координатор прогона
- `common/runner.py` — TestRunner (замена pytest)

---

*Last updated: 2026-06-22 · Кодо*
