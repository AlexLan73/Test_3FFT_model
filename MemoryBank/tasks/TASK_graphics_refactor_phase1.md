# 🧩 TASK — graphics_refactor · phase1

> **Спека:** [`specs/graphics_refactor_2026-07-06.md`](../specs/graphics_refactor_2026-07-06.md)
> **Статус:** ✅✅ РЕАЛИЗОВАНО + ОТРЕВЬЮВАНО (Sonnet → Кодо, 2026-07-07). Все 7 шагов зелёные:
> `mypy core/` 0 ошибок, `ruff check core/` чист, `tests/all_test.py` 10 ok, `main.py`/`demo_interactive.py` рисуют PNG/HTML. plotly не течёт в статик-ветку. mypy 2.1.0 доустановлен, `[tool.mypy] python_version` → 3.12.
> Правки по ревью F1–F7 внесены в шаги 1–6 (пометки ⚠️/F#).
> **Ветка абстракций:** matplotlib (есть) + **plotly параллельно** (новое, не смешивать).
> **Критично:** 🚫 pytest (только `common.runner.TestRunner`) · 🚫 писать в `.claude/worktrees/**`.

---

## 🎯 Цель

Перенести черновые графики (`Example/Hener1/files 8/scenario_v2_*.py`) в `core/graphics/`
как параметризованные стратегии; добавить параллельную интерактивную ветку (plotly);
воспроизвести текущие картинки из архитектуры. Данные — по ссылке (`SpectralCube`),
вывод — абстрактный (render отделён от write).

---

## 📋 Шаги (порядок строгий, коммит после каждого зелёного шага)

### Шаг 1 — `AxisLayout` (VO) + `CubeSampler` (Pure Fabrication)  ⬜
- `core/graphics/layout.py`: `AxisLayout` (frozen dataclass) + `range_vertical()` / `range_in_depth()`
  + хелпер `resolve(cube, axis_key) -> (values, label, limits)`. **label** — из карты `axis_key→label`
  внутри layout (у `Axis` label НЕТ, только техн. `name`); **limits** — `values.min()/max()`
  (у `Axis` `limits` НЕТ — вычислять), для `Axis.centered=True` симметричный паддинг ±0.5..1.
- `core/graphics/sampling.py`: `CubeSampler(threshold_db, range_limit)` →
  `points(cube, layout) -> SampledPoints(x,y,z,values_db,mask)`. Только чтение куба, без мутаций.
  Порядок: `m = cube.magnitude_db` (**глобальный** max, НЕ пересчитывать после обрезки)
  → срез `range_limit` по **axis=2** куба → `mask = m > threshold_db` → meshgrid по раскладке.
- Тест: `tests/test_graphics.py::GraphicsTests.test_layout_axes`, `test_sampler_threshold`.
- **DoD:** маска = `values_db > threshold`; оси берутся из `cube.kx/ky/range` по раскладке;
  `range_limit` режет дальность куба независимо от того, куда `range` ложится на экране.

### Шаг 2 — параметризовать `CubeScatterVisualizer(layout=…)`  ⬜
- Дефолт `layout=AxisLayout.range_vertical()` → **байт-в-байт прежний вид** (нет регрессии).
- Рендер через `CubeSampler` (убрать дублируемый meshgrid/threshold-код).
- Метки/пределы осей — из раскладки, не хардкод.
- Тест: `test_scatter_default_regression`, `test_scatter_range_in_depth`.
- **DoD:** `python main.py` даёт визуально прежний `cube_scatter.png` при `range_vertical()`;
  регрессию проверяем по **набору точек/маске** (`CubeSampler`), НЕ по байтам PNG (matplotlib
  байт-идентичность не гарантирует). Новая раскладка `range_in_depth()` — валидная Figure.
  ⚠️ дефолт визуализатора `threshold_db=-20`, а `main.py` передаёт `-22` — при рефакторинге
  сохранить оба (не хардкодить порог внутри рендера).

### Шаг 3 — подпакет `interactive/` (plotly, параллельно)  ⬜
- `interactive/interactive_visualizer.py`: `InteractiveVisualizer(ABC).render(cube) -> go.Figure`.
- `interactive/cube_interactive.py`: `InteractiveCubeVisualizer(layout, thresholds, default_db)`;
  слайдер порога = кадры видимости; вращение — нативное plotly; переиспользует `CubeSampler`.
  ⚠️ гард: если `default_db not in thresholds` — брать ближайший (`min(thresholds, key=...)`),
  иначе `thresholds.index(default_db)` кинет `ValueError` (латентный баг черновика, F6).
- `interactive/html_writer.py`: `HtmlWriter.write(fig,name)` → `write_html(include_plotlyjs=True)`.
- **Импорт plotly — только внутри `interactive/`** (matplotlib-ветка от plotly не зависит):
  `import plotly` — на уровне модулей подпакета, НЕ в `core/graphics/__init__.py`.
- Тест: `test_interactive_html_written` (файл создан, размер > 0, есть `<div` plotly) — под `SkipTest`,
  если plotly не установлен.

### Шаг 4 — реэкспорт (две ветки раздельно)  ⬜
- `core/graphics/__init__.py`: добавить **только** `AxisLayout`, `CubeSampler` (+ прежние
  matplotlib-имена). **НЕ** реэкспортить plotly-имена сюда (F3 — иначе `import core.graphics`
  тянет plotly, ломает мягкую зависимость и критерий §9).
- `core/graphics/interactive/__init__.py`: реэкспорт `InteractiveVisualizer`,
  `InteractiveCubeVisualizer`, `HtmlWriter`. plotly-имена доступны через `core.graphics.interactive`.

### Шаг 5 — Composition Root  ⬜
- `main.py`: добавить job `cube_scatter_range_depth.png` (новая раскладка) рядом с текущими.
- `demo_interactive.py` (корень, новый): собрать `InteractiveCubeVisualizer` + `HtmlWriter`,
  прогнать `default_scenario()` → `out/figures/cube_interactive.html`.

### Шаг 6 — тесты (TestRunner, БЕЗ pytest)  ⬜
- `tests/test_graphics.py` — набор `GraphicsTests(TestRunner)` со `SkipTest` если plotly нет.
  API раннера: `AssertionGroup("graphics.xxx")` + `g.add(cond, msg)`, метод `test_*` → возвращает group.
- `tests/all_test.py` **уже существует** (F7): `SUITES = [SmokeTests]` → дописать `GraphicsTests`
  в список и импорт. НЕ создавать файл заново.

### Шаг 7 — зависимости + чистка  ⬜
- `pyproject.toml`: optional-dependency `viz = ["plotly>=6"]`.
- Перенести/удалить `Example/Hener1/files 8/scenario_v2_axes.py`, `scenario_v2_interactive.py`,
  `scen_cube_v2.png`, `scen_cube_interactive.html` (архив-эталон `scen_cube.png` — оставить).
- `ruff check core/ && mypy core/` — чисто.

---

## ✅ Definition of Done (вся фаза)

- [ ] `python main.py` — без регрессии + новая раскладка осей.
- [ ] `python demo_interactive.py` — валидный интерактивный HTML.
- [ ] `python tests/test_graphics.py` — зелено (TestRunner).
- [ ] `ruff` + `mypy` — чисто.
- [ ] matplotlib-ветка не импортирует plotly.
- [ ] нет новых скриптов в `Example/**`.

---

## 🔍 Проверка за Sonnet (делает Кодо)

1. Диф по §критериям приёмки спеки §9.
2. Прогнать все 3 команды запуска, глазами сверить PNG/HTML.
3. Проверить: `AxisLayout` дефолт не изменил старый вид; plotly не протёк в статик-ветку.
4. Отчёт → `MemoryBank/sessions/<дата>.md` + отметить шаги ✅ здесь.

---

## ❓ Перед стартом — решить (см. спеку §8)

Pipeline вводим? · `experiments/` нужен? · plotly в офлайн-пак? · kaleido? · судьба `Example/`.
