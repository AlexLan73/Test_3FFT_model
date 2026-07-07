# 🧩 TASK — anti-barrage · CA-CFAR детектор · phase1

> **Спека:** [`specs/anti_barrage_detection_2026-06-23.md`](../specs/anti_barrage_detection_2026-06-23.md) §2.4
> **Мат-модель:** `Doc/anti_barrage_math.md` §7.
> **Статус:** ✅✅ РЕАЛИЗОВАНО + ОТРЕВЬЮВАНО (Sonnet-код + Кодо-фикс, 2026-07-07).
> Ревью вскрыло self-masking (широкий мейнлоуб n_real=16): опорные ячейки садились на юбку цели
> → цель не детектировалась. Фикс Кодо: `n_guard=4` (перекрывает мейнлоуб) + `local_max` гейт
> по дальности (схлопывает мейнлоуб в 1 отсчёт). Цель найдена, all_test 21 ok, mypy/ruff чисто.
> Остаток: ~5 ложных в столбе barrage (остаток нуллинга 23.9 дБ) + угловой мейнлоуб цели → phase2.
> (Sonnet-агент дважды падал по обрыву связи — тест и фикс дописал Кодо.)
> **Критично:** 🚫 pytest (только `common.runner.TestRunner`) · 🚫 писать в `.claude/worktrees/**`.
> **Неинвазивно:** контроллер и существующий тракт НЕ трогать. Только аддитивно.

---

## 🎯 Цель

Детерминированный **CA-CFAR** детектор по оси дальности: после SubspaceNuller замкнуть
цепочку «подавили помеху → обнаружили пик цели» с заданной вероятностью ложной тревоги.
Локализованный пик цели проходит порог; остаток заливки (barrage) — нет.

## 📐 Математика (Doc/anti_barrage_math.md §7)

- Тестовая ячейка `CUT` по оси дальности; вокруг — `n_guard` защитных + `n_train` опорных
  с каждой стороны. `N = 2·n_train` опорных ячеек.
- Оценка шума `P̂ = (1/N) Σ Pᵢ` по опорным (мощность `P = |C|²`).
- Порог `T = α·P̂`, `α = N·(P_fa^(−1/N) − 1)`.
- Решение «цель», если `P_CUT > T`.

## 📋 Шаги

### Шаг 1 — `core/models/anti_barrage/cfar.py`  ⬜
- `Detection` — frozen dataclass (VO): `kx_idx, ky_idx, range_bin: int`, `level_db, threshold_db: float`,
  `kx, ky: float` (углы из осей куба).
- `CaCfarDetector`:
  - `__init__(self, pfa: float = 1e-3, n_train: int = 8, n_guard: int = 2)` — валидация >0.
  - `alpha` — свойство/поле: `n*(pfa**(-1/n) − 1)`, `n = 2*n_train`.
  - `detect(self, cube: SpectralCube) -> list[Detection]` — по каждой угловой ячейке `(ix,iy)`
    скользящий CA-CFAR вдоль оси range (axis=2). Края (где окно не помещается) — пропускать
    или усекать опорную выборку (документировать выбор). Только ЧТЕНИЕ куба (magnitude), без мутаций.
  - `detect_cell(self, cube, ix, iy) -> list[Detection]` — CFAR в одной угловой ячейке (для тестов/гейта).
  - Работать на `P = cube.magnitude**2`; `level_db`/`threshold_db` — 10·log10 отн. глобального max.
- `__init__.py` подпакета: добавить реэкспорт `CaCfarDetector`, `Detection`.
- Прочитать РЕАЛЬНЫЙ `core/models/result.py` (SpectralCube API: `magnitude`, оси `kx/ky/range`,
  `index_of_angle`) — не гадать.

### Шаг 2 — демо `demo_cfar.py` (корень)  ⬜
- Сцена target+barrage под разными углами → `Synthesizer` → `SubspaceNuller().apply` →
  `Fft3DModel.process` → `CaCfarDetector().detect`.
- Напечатать: найденные детекции (угол, бин дальности, level/threshold дБ); подтвердить, что цель
  найдена в правильной ячейке, а в столбе barrage ложных тревог нет (или единицы при заданном P_fa).
- Сравнить детекции ДО и ПОСЛЕ нуллинга (до — заливка мешает, после — чистый пик).

### Шаг 3 — тест `tests/test_cfar.py` (TestRunner, БЕЗ pytest)  ⬜
- `CfarTests(TestRunner)` + `AssertionGroup`:
  - target-сцена: CFAR находит цель в ожидаемом бине дальности (±1) в её угловой ячейке.
  - формула порога: `alpha` совпадает с `N(P_fa^(−1/N)−1)` численно.
  - монотонность: меньший `P_fa` → больший порог → не больше детекций.
  - чистый шум (thermal-only): число ложных тревог мало (≈ P_fa · число ячеек, с запасом).
  - full-chain: barrage+target → nuller → CFAR → цель найдена, в столбе помехи ложных нет.
- Дописать `CfarTests` в `tests/all_test.py` SUITES (файл есть — не создавать заново).

## ✅ Definition of Done

- [ ] `python demo_cfar.py` — цель детектируется, ложных в barrage нет, до/после нуллинга видно.
- [ ] `python tests/all_test.py` — всё зелено (Smoke + Graphics + Nuller + Cfar).
- [ ] `mypy core/` 0 ошибок · `ruff check core/` чисто.
- [ ] Контроллер/существующий тракт НЕ изменён (`git diff` — только новые файлы + `__init__.py` + all_test.py).
- [ ] `detect` не мутирует куб.

## 🔮 Phase2 (НЕ сейчас)

OS-CFAR/GO-CFAR варианты, 2D-CFAR по углу+дальности, torch/GPU-бэкенд, впайка в pipeline;
далее Доплер (§5) → 3D-CNN (§8).
