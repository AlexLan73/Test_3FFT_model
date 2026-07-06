# 📐 Спека — рефакторинг графики radar3d (2026-07-06)

> **Автор:** Кодо · **Статус:** ✅ ОТРЕВЬЮВАНА (Кодо, 2026-07-06) → готова к реализации Sonnet
>
> ### 🔍 Правки по ревью (сверка с реальным кодом)
> - **F1** §5/§9: черновики = куб **16×16×256** (ZMAX=130), а `default_scenario()` = **16×16×64**
>   (`n_fft=64`). Миграция воспроизводит **технику** (раскладка/слайдер), а НЕ пиксели черновика.
>   Пиксельная сверка `scen_cube_v2.png` невозможна — черновики остаются архив-эталоном.
> - **F2** §3.1: `Axis` = `name/values/centered` (нет `label`/`limits`). Хелпер берёт метки из
>   карты `axis_key→label` в layout, лимиты — из `values.min()/max()` (+паддинг для угловых осей).
> - **F3** §2/§4: plotly-имена реэкспортим **только** в `interactive/__init__.py`, НЕ в
>   `core/graphics/__init__.py` — иначе `import core.graphics` тянет plotly (ломает мягкую зависимость).
> - **F4** §3.2: `CubeSampler` нормирует по `cube.magnitude_db` (**глобальный** max, Information Expert),
>   потом режет `range_limit` и порог. Отличается от черновика (локальный max по `[:130]`) — это норма.
> - **F5** §9: критерий «байт-в-байт PNG» снят — сравниваем **набор точек (маску)**, не байты файла.
> **Тема:** перенести черновые визуализации (`Example/Hener1/**`) в архитектуру `core/graphics/`,
> причесать под единый стиль, добавить **параллельную** интерактивную ветку (plotly).
> **Инициатор:** Alex (5 требований, см. ниже).

---

## 0. Требования Alex (дословный разбор)

| № | Требование | Как закрываем |
|---|-----------|---------------|
| 1 | «Писать в стиле ООП/SOLID/GRASP/GoF в наших каталогах; где мне это зафиксировать» | новое правило `.claude/rules/06-graphics.md` + раздел §7 этой спеки |
| 2 | «Спец-класс графики: ограниченный набор вариантов, рисуют ~одно и то же — причесать» | уже есть `Visualizer(ABC)`; причёсываем через `AxisLayout` (VO) + базовые хелперы `_CubeSampler` |
| 3 | «Писать в заложенной архитектуре; добавить/убавить — обсудить» | встраиваемся в `core/graphics/`, **matplotlib-ветку не ломаем**; изменения только аддитивные |
| 4 | «Перенести все работы + графики (сделать как сейчас)» | §5 — таблица миграции: v2-оси + plotly-интерактив → стратегии в `core/graphics/`, вывод в `out/` |
| 5 | «Вывод графики абстрактный, данные передавать по ссылке» | **уже так**: `render(cube: SpectralCube)`; формализуем протоколами (§4) |

**Развилка (решена Alex):** `render()` в matplotlib-ветке возвращает `matplotlib.Figure`,
а plotly — свой `plotly.graph_objects.Figure`. Alex: *«создай параллельно под другое решение»* →
делаем **две независимые ветки абстракций** (ISP), НЕ обобщаем один `render` под оба бэкенда.

---

## 1. Что уже есть (не трогаем, база)

```
core/graphics/
  visualizer.py     Visualizer(ABC).render(cube: SpectralCube) -> matplotlib.Figure   # Strategy
  cube_scatter.py   CubeScatterVisualizer         # 3D-скаттер куба
  angular_map.py    AngularMapVisualizer          # угловая карта энергии + гейт
  range_profile.py  RangeProfileVisualizer        # дальностные профили ячеек
  writer.py         FigureWriter.write(fig, name) # Pure Fabrication (PNG на диск)
  __init__.py       реэкспорт
```

`SpectralCube` ([core/models/result.py](../../core/models/result.py)) — Value Object,
Information Expert: `magnitude`, `magnitude_db`, `angular_energy_db()`,
`range_profile_db(ix,iy)`, `index_of_angle(kx,ky)`, оси `kx/ky/range: Axis`.
**Данные уже ходят по ссылке** — визуализатор получает объект куба, ничего не копирует.

Оценка: архитектура здоровая. Работы Alex (`scen_cube_v2`, plotly) — это **новые/параметризованные
стратегии**, а не переписывание.

---

## 2. Целевая структура `core/graphics/`

```
core/graphics/
  layout.py         AxisLayout (VO)               # НОВОЕ: раскладка осей куба -> X/Y/Z графика
  sampling.py       CubeSampler (Pure Fabrication)# НОВОЕ: порог/выборка точек из куба (общий код)
  visualizer.py     Visualizer(ABC)               # без изменений (matplotlib-ветка)
  cube_scatter.py   CubeScatterVisualizer(layout=…)# ПАРАМЕТРИЗУЕМ раскладкой осей
  angular_map.py    AngularMapVisualizer          # без изменений
  range_profile.py  RangeProfileVisualizer        # без изменений
  writer.py         FigureWriter                  # без изменений (PNG)

  interactive/                                     # НОВАЯ ПАРАЛЛЕЛЬНАЯ ВЕТКА (plotly)
    __init__.py
    interactive_visualizer.py  InteractiveVisualizer(ABC).render(cube) -> plotly Figure
    cube_interactive.py        InteractiveCubeVisualizer(layout=…, thresholds=…)
    html_writer.py             HtmlWriter.write(fig, name) -> path   # fig.write_html

  __init__.py       реэкспорт ТОЛЬКО matplotlib-ветки (+ AxisLayout, CubeSampler)
```

> plotly вынесен в подпакет `interactive/`, чтобы **matplotlib-ветка не зависела от plotly**
> (импорт plotly только при использовании интерактива — мягкая зависимость).
>
> ⚠️ **F3 (критично):** `core/graphics/__init__.py` реэкспортит `AxisLayout`, `CubeSampler` и
> matplotlib-визуализаторы, но **НЕ** plotly-имена. `InteractiveVisualizer`/`InteractiveCubeVisualizer`/
> `HtmlWriter` реэкспортим **только из `interactive/__init__.py`**. Иначе `import core.graphics`
> потянет `from .interactive... import ...` → plotly импортируется всегда → нарушен критерий §9
> «matplotlib-ветка не импортирует plotly». (Альтернатива — ленивый `__getattr__`, но проще не тянуть.)

---

## 3. Новые сущности (минимум, без раздувания)

### 3.1 `AxisLayout` — Value Object (закрывает п.2 «причесать»)

Одна стратегия скаттера рисует **обе** раскладки без дубля класса.

```python
# core/graphics/layout.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AxisLayout:
    """Куда ложатся оси куба на график. axis_x/y/z ∈ {'kx','ky','range'}."""
    axis_x: str = "kx"
    axis_y: str = "ky"
    axis_z: str = "range"

    @classmethod
    def range_vertical(cls) -> "AxisLayout":       # текущий вид (дальность — вверх)
        return cls("kx", "ky", "range")

    @classmethod
    def range_in_depth(cls) -> "AxisLayout":       # v2: дальность в глубину, ky вертикаль
        return cls("kx", "range", "ky")
```

Хелпер `resolve(cube, axis_key) -> (values, label, limits)`:
- `values` — из `cube.kx/ky/range` по ключу (`Axis.values`);
- `label` — из карты в `AxisLayout`: `{"kx":"kx (азимут)", "ky":"ky (угол места)", "range":"дальность (≥0)"}`
  (⚠️ `Axis` НЕ хранит человекочитаемый label — только технический `name`);
- `limits` — `(values.min(), values.max())`; для центрированных осей (`Axis.centered=True`) симметричный
  паддинг `±0.5..1` как в черновиках (`set_xlim(-8, 8)`). `Axis.limits` НЕ существует — вычисляем.

### 3.2 `CubeSampler` — Pure Fabrication (общий код обеих веток)

Порог + выборка точек > threshold_db одинаковы у matplotlib и plotly — выносим:

```python
# core/graphics/sampling.py
class CubeSampler:
    def __init__(self, threshold_db: float, range_limit: int | None = None): ...
    def points(self, cube: SpectralCube, layout: AxisLayout) -> SampledPoints: ...
    # SampledPoints: x, y, z (np.ndarray по раскладке), values_db, mask
```

**Нормировка и порядок (важно, F4):**
1. `m = cube.magnitude_db` — нормировка по **глобальному** max всего куба (Information Expert,
   так уже делает `CubeScatterVisualizer`). НЕ пересчитывать max после обрезки.
2. `range_limit` режет **дальностную ось куба (axis=2)** ДО meshgrid — независимо от того, на какую
   экранную ось (X/Y/Z) `range` ложится по раскладке.
3. `mask = values_db > threshold_db` (строго `>`, как в текущем коде).
4. `x/y/z` — meshgrid по раскладке `layout`, `indexing="ij"`, `.ravel()[mask]`. Куб не мутируется.

> ⚠️ Черновики нормируют по локальному max (`m = 20*log10(mag[:,:, :ZMAX]); m -= m.max()`, eps 1e-9).
> Архитектура — по глобальному (eps 1e-12). Набор точек при пороге -22 дБ будет отличаться — это ОК
> (для `default_scenario()` max лежит внутри диапазона, расхождение мало́).

### 3.3 Интерактивная ветка (plotly) — параллельная абстракция

```python
# core/graphics/interactive/interactive_visualizer.py
class InteractiveVisualizer(ABC):
    @abstractmethod
    def render(self, cube: SpectralCube) -> "plotly.graph_objects.Figure": ...

# core/graphics/interactive/cube_interactive.py
class InteractiveCubeVisualizer(InteractiveVisualizer):
    def __init__(self, layout: AxisLayout = AxisLayout.range_in_depth(),
                 thresholds=range(-40,-5,2), default_db=-22): ...
    # слайдер порога = кадры видимости по thresholds; крутить мышью — нативно plotly

# core/graphics/interactive/html_writer.py
class HtmlWriter:
    def __init__(self, out_dir: str): ...
    def write(self, fig, name: str) -> str:   # fig.write_html(include_plotlyjs=True)
```

---

## 4. Формализация «вывод абстрактный + данные по ссылке» (п.5)

Два симметричных протокола (ISP), обе ветки им следуют:

```
Static family:        Visualizer.render(cube) -> mpl.Figure     →  FigureWriter.write(fig,name)
Interactive family:   InteractiveVisualizer.render(cube) -> plotly.Figure →  HtmlWriter.write(fig,name)
```

- **Данные по ссылке:** оба `render(cube)` получают один и тот же объект `SpectralCube`,
  ничего не мутируют, не копируют массивы (только читают/срезают view).
- **Абстрактный вывод:** рендер отделён от записи; куда/чем писать — решает Composition Root.
- **OCP:** новый визуализатор = подкласс + строка в связывании `main.py`; ни ветку, ни writer не трогаем.

Опционально (обсудить, §8) — тонкий `GraphicsPipeline`, чтобы `main.py` не держал два словаря.

---

## 5. Миграция текущих работ (п.4 «сделать как сейчас»)

| Черновик (`Example/Hener1/files 8/`) | Куда переезжает | Результат |
|--------------------------------------|-----------------|-----------|
| `scenario_v2_axes.py` (перестановка осей, PNG) | `CubeScatterVisualizer(layout=AxisLayout.range_in_depth())` | `out/figures/cube_scatter_range_depth.png` |
| `scenario_v2_interactive.py` (plotly + слайдер) | `InteractiveCubeVisualizer` + `HtmlWriter` | `out/figures/cube_interactive.html` |
| `scen_cube_v2.png`, `scen_cube_interactive.html` | генерятся из `core/` через демо/main | техника воспроизводима* |
| исходные PNG `scen_cube.png/scen_filter.png` + `scenario.py` | **оставить как архив-эталон** для сверки | не трогаем |

> \* **F1 (важно):** черновики стоят на **standalone**-данных **16×16×256** (`Lr=256`, `ZMAX=130`,
> источники на бинах 40/60/80/100). Архитектурный `default_scenario()` даёт куб **16×16×64**
> (`n_fft=64`, DRFM бины 8,14,20,26,32). Поэтому миграция воспроизводит **приём** (перестановка осей,
> plotly-слайдер), а НЕ пиксели `scen_cube_v2.png`. Пиксельная сверка невозможна и не требуется —
> `Example/Hener1` остаётся архив-эталоном. Если понадобится именно 256-бинная картинка —
> завести отдельный демо-сценарий `RangeConfig(n_fft=256)` + источники черновика (не в scope phase1).
> Дефолт `default_db=-22` и `thresholds=range(-40,-5,2)` подобраны под 256-бинную сцену — на 64-бинном
> кубе проверить визуально, при необходимости подстроить в `demo_interactive.py`.

Запуск (как сейчас, но из архитектуры):
```bash
python main.py            # статичные PNG (в т.ч. новая раскладка осей)
python demo_interactive.py# интерактивный HTML (plotly)   ← новый тонкий демо-скрипт в корне
```

---

## 6. План работ (для таска — краткая версия, детали в TASK-файле)

1. `layout.py` + `sampling.py` (VO + Pure Fabrication) + тест `TestRunner`.
2. Параметризовать `CubeScatterVisualizer` через `AxisLayout` (дефолт = текущий вид → без регрессии).
3. Подпакет `interactive/`: `InteractiveVisualizer`, `InteractiveCubeVisualizer`, `HtmlWriter`.
4. Реэкспорт в `core/graphics/__init__.py` (обе ветки).
5. `main.py`: добавить job новой раскладки; `demo_interactive.py` — plotly HTML.
6. Тесты: `tests/test_graphics.py` (shape/threshold/layout/файл создан) — **только TestRunner, без pytest**.
7. Зависимости: `pyproject.toml` extra `[viz] = plotly`; §8 — офлайн-колесо для дома.
8. Почистить черновики после переноса (оставив архив-эталон §5).

---

## 7. Правило стиля для графики (ответ на п.1 «где мне написать»)

Создать `.claude/rules/06-graphics.md` (paths: `core/graphics/**`) — короткое:

- Любой новый график = **подкласс** `Visualizer` (статик) **или** `InteractiveVisualizer` (интерактив).
- `render(cube)` только **читает** `SpectralCube` (по ссылке), НЕ мутирует, НЕ пишет на диск.
- Запись — только через `FigureWriter`/`HtmlWriter` (IO отделён).
- Общий код (порог, выборка, раскладка осей) — в `sampling.py`/`layout.py`, не копипастить.
- Черновые эксперименты с графикой — **сразу как стратегия** в `core/graphics/`, запуск через
  `demo_*.py` в корне; **не** плодить скрипты в `Example/**` (это архив-эталоны, read-only).
- matplotlib и plotly — **параллельны**, не смешивать в одном классе.

> Так на вопрос «где писать, чтобы было в стиле» ответ: **в `core/graphics/` по правилу 06**,
> а не отдельными скриптами в `Example/`.

---

## 8. Открытые вопросы + рекомендации ревью (Кодо, 2026-07-06)

1. **`GraphicsPipeline`** — **рекомендую НЕ вводить в phase1.** Явные словари в `main.py` уже читаемы
   (Composition Root), +1 сущность не окупается на 4 визуализаторах. Ввести позже, если веток станет >6.
2. **Каталог черновиков** — **сразу `core/` + `demo_*.py`** (меньше сущностей, согласуется с правилом 06).
   `experiments/` не заводим.
3. **plotly в офлайн-пак** — **да, добавить колесо** `plotly` + `narwhals` + `packaging` в
   `/mnt/data/offline-debian-pack/3_python_wheels/` (иначе дома без сети `viz`-extra не поставится).
   ⚠️ не качать без явного OK Alex (правило: тяжёлые скачивания — по спросу).
4. **kaleido (PNG-экспорт plotly)** — **НЕ тащим** (тяжёлый, тянет chromium). HTML самодостаточен.
5. **Судьба `Example/Hener1`** — **оставить как архив-эталон** (§5, F1): черновики стоят на 256-бинных
   standalone-данных, из `default_scenario()` (64 бина) их пиксельно не воспроизвести. Удаляем только
   `.py`-черновики после переноса техники; PNG/HTML-эталоны остаются read-only для истории.

## ответ Alex
1. Согласен НЕ вводить в phase1 - Да
2. Да
3. Да
4. Да
5. Да

---

## 9. Критерии приёмки

- `python main.py` — без ошибок, PNG прежние **плюс** новая раскладка осей; регрессии нет
  (дефолтный `AxisLayout.range_vertical()` = текущий вид). Сверка регрессии — по **набору точек/маске**
  (`CubeSampler`), НЕ по байтам PNG (F5).
- `python demo_interactive.py` — создаёт валидный самодостаточный `cube_interactive.html`
  (крутится мышью, слайдер порога работает). На данных `default_scenario()` (64 бина, не 256) — картинка
  отличается от `scen_cube_interactive.html`, это ожидаемо (F1).
- `python tests/test_graphics.py` — все группы зелёные (TestRunner, без pytest; `SkipTest` если нет plotly).
- `ruff check core/ && mypy core/` — чисто.
- matplotlib-ветка **не импортирует** plotly: проверка `import core.graphics` без установленного plotly
  проходит (plotly-имена доступны только через `core.graphics.interactive`, F3).
- Ни одного нового скрипта в `Example/**`; черновики перенесены/удалены.

---

*Ревью → правки → реализация Sonnet по §6/TASK → Кодо проверяет за Sonnet по §9.*
