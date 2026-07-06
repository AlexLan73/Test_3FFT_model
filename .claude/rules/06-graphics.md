# 06 — Graphics (правила модуля визуализации)

> **paths:** `core/graphics/**`, `demo_*.py`
> Ответ на вопрос «где писать графику, чтобы было в стиле ООП/SOLID/GoF».

## 🎯 Главное правило

**Любой новый график = подкласс стратегии, а не отдельный скрипт.**
Не плодить `.py` в `Example/**` — там только архив-эталоны (read-only).

## Две параллельные ветки (не смешивать!)

| Ветка | Абстракция (Strategy) | Возвращает | Запись (Pure Fabrication) |
|-------|----------------------|-----------|---------------------------|
| статик (matplotlib) | `Visualizer(ABC).render(cube)` | `matplotlib.Figure` | `FigureWriter` → PNG |
| интерактив (plotly) | `InteractiveVisualizer(ABC).render(cube)` | `plotly.go.Figure` | `HtmlWriter` → HTML |

- `matplotlib`-ветка **НЕ импортирует** plotly. plotly живёт только в `core/graphics/interactive/`.
- `core/graphics/__init__.py` реэкспортит статик-имена + `AxisLayout`/`CubeSampler`,
  **но не** plotly-имена (иначе `import core.graphics` тянет plotly). Интерактив —
  через `core.graphics.interactive`.

## Контракт `render(cube)`

- Принимает `SpectralCube` **по ссылке**, только **читает** (`magnitude_db`, оси `kx/ky/range`).
- **НЕ мутирует** куб, **НЕ пишет** на диск (запись — дело writer'а, в Composition Root).
- Куда/чем сохранять — решает `main.py`/`demo_*.py`, не сам визуализатор.

## Не дублировать общий код

- Порог + выборка точек → `CubeSampler` (нормировка по **глобальному** `magnitude_db`,
  срез `range_limit` по дальностной оси куба, `mask = db > threshold`).
- Раскладка осей куба на экран → `AxisLayout` (VO): одна стратегия рисует разные виды
  (`range_vertical()` / `range_in_depth()`), а не N почти одинаковых классов.
- Человекочитаемые подписи/лимиты осей — в `AxisLayout` (у `Axis` только техн. `name`).

## Запуск / связывание

- Стратегии связываются с writer'ами в **Composition Root** (`main.py` — статик,
  `demo_interactive.py` — plotly). Без глобалов.
- Тесты графики — `tests/test_graphics.py` через `TestRunner` (🚫 pytest, см. правило 04);
  plotly-тесты под `SkipTest`, если библиотеки нет.

## Зависимости

- plotly — **опциональная** (`pyproject.toml [project.optional-dependencies] viz`).
  Базовый тракт (статик-графика) работает без plotly.
