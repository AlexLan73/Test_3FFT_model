# 🤖 CLAUDE — Test_3FFT_model (radar3d)

> **Проект**: radar3d — пространственно-временна́я **3D-БПФ** обработка для
> предварительной сортировки сигналов на матричных ядрах GPU (каскад 1–2 заявки)
> + каскад-классификатор (RuleBased → 3D-CNN).
> **Платформа**: Debian Linux (работа) + Windows (дома). Python ≥ 3.11.
> **GPU (опц.)**: PyTorch-ROCm для обучаемого классификатора (AMD).
> **Ассистент**: Кодо (Claude)
> **Repo**: https://github.com/AlexLan73/Test_3FFT_model

---

## 🧠 Режим работы ассистента

Модульные правила проекта → **`.claude/rules/*.md`** (6 файлов).
Конфиг гармонизирован с базовыми проектами `DSP-GPU` и `rag-mentor`.

---

## 👤 Alex

- Обращаться к Кодо: «**Любимая умная девочка**» или «**Кодо**» (мужчина, senior).
- Кодо обращаться к Alex: «**Alex**».
- Русский, неформально, с эмодзи — **по делу**.
- Детали → `.claude/rules/01-user-profile.md`.

---

## 🚨 2 критических правила (нарушать нельзя)

1. **🚫 pytest ЗАПРЕЩЁН** — только `common.runner.TestRunner` + `SkipTest`.
   (Прецедент DSP-GPU: потеряны дни работы.) → `.claude/rules/04-testing-python.md`
2. **🚨 НЕ писать в `.claude/worktrees/*/`** — файлы теряются, не попадают в git.
   → `.claude/rules/03-worktree-safety.md`

---

## 🏗️ Архитектура

Слои с **инверсией зависимостей**, связывание в `main.py` (Composition Root).

| каталог (код)        | назначение                       |
|----------------------|----------------------------------|
| `core/config/`       | настройки модели (Value Object)  |
| `core/data_context/` | load/save (Facade + Repository)  |
| `core/generators/`   | сигналы + помехи (Factory/Builder) |
| `core/graphics/`     | визуализаторы (Strategy)         |
| `core/models/`       | 3FFT и далее (Template Method)   |
| `core/models/classification/` | RuleBased + 3D-CNN (Strategy/LSP) |
| `core/controller.py` | координатор прогона               |

**Паттерны GoF**: Strategy, Composite, Abstract Factory/Registry, Builder,
Template Method, Facade, Value Object, Pure Fabrication, Dependency Injection.

> ⚠️ Имена пакетов **строчные ASCII** (PEP 8): `core`, не `Core`. На Linux ФС
> регистрозависима — заглавная папка ломает импорты.

Стиль кода → `.claude/rules/05-python-style.md`.

---

## 🧪 Классы сцены

`empty`, `target`, `barrage` (заградительная), `comb` (гребёнка), `ham` (радиолюбитель).

- `RuleBasedClassifier` — детерминированный, без torch, доказуем.
- `Cnn3DClassifier` — обучаемая 3D-CNN (PyTorch-ROCm), та же абстракция (LSP).

---

## 🚀 Запуск

```bash
python main.py            # эталонный прогон -> ./out/figures, ./out/data
python classify_demo.py   # детерминированная классификация (без torch)
python train_cnn.py --steps 400 --batch 40   # обучение -> cnn3d.pt (нужен torch+ROCm)
```

Установка зависимостей:
```bash
# venv ДОЛЖЕН быть Python 3.12 (колёса torch-ROCm — cp312, не 3.13!):
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .            # numpy + matplotlib + scipy

# torch+ROCm 7.2 — из офлайн-пака (cp312), triton не нужен (только для torch.compile):
PACK=/mnt/data/offline-debian-pack/3_python_wheels
.venv/bin/python -m pip install --no-deps \
  "$PACK/torch-rocm/torch-2.11.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl"
```
> ✅ Проверено: torch 2.11.0+rocm7.2, GPU **AMD Radeon RX 9070** (gfx1201),
> `torch.cuda.is_available()=True`, CNN обучается на `device=cuda`.

---

## 🚀 Новая задача — обязательная последовательность

```
сформулировать вопрос
  → Context7 MCP (доки библиотек: numpy/scipy/torch/matplotlib)
  → WebFetch/URL (свежие статьи)
  → sequential-thinking MCP (если сложная — архитектура/математика)
  → GitHub MCP (референсный код)
  → ТОЛЬКО теперь писать код
```

Детали → `.claude/rules/00-new-task-workflow.md`.

---

## 🗣️ Команды Alex

| Команда | Действие |
|---------|---------|
| «Покажи статус» | `MemoryBank/MASTER_INDEX.md` + `MemoryBank/tasks/IN_PROGRESS.md` |
| «Добавь задачу: ...» | `MemoryBank/tasks/TASK_<topic>_<phase>.md` |
| «Запиши в спеку: ...» | `MemoryBank/specs/{topic}_YYYY-MM-DD.md` |
| «Что сделали сегодня?» | `MemoryBank/sessions/YYYY-MM-DD.md` |

---

## 🎯 Приоритеты

1. ✅ **Работоспособность** — главное, чтобы работало.
2. 🎯 **Корректность** — сверка с эталоном (NumPy / SciPy).
3. ⚡ **Производительность** — векторизация, потом GPU.
4. 📝 **Документация** — после стабилизации API.
5. 🧹 **Очистка** — удалить промежуточные файлы.

---

*Last updated: 2026-06-22 · Maintained by: Кодо · Source: DSP-GPU + rag-mentor configs*
