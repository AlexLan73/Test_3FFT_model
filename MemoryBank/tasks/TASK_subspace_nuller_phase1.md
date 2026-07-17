# 🧩 TASK — anti-barrage · SubspaceNuller · phase1

> **Спека:** [`specs/anti_barrage_detection_2026-06-23.md`](../specs/anti_barrage_detection_2026-06-23.md)
> **Мат-модель:** `Doc/anti_barrage_math.md` §3–4.
> **Статус:** ✅✅ РЕАЛИЗОВАНО + ОТРЕВЬЮВАНО (Sonnet → Кодо, 2026-07-07). barrage −20.3 дБ,
> цель kx=+2 −7.8→0 дБ (доминанта), is_barrage True/False верно. mypy/ruff чисто, all_test 15 ok
> (Smoke 3 + Graphics 7 + Nuller 5), тракт/контроллер не тронуты. Риски порогов/rank-deficient → phase2.
> **Критично:** 🚫 pytest (только `common.runner.TestRunner`) · 🚫 писать в `.claude/worktrees/**`.
> **Неинвазивно:** контроллер и существующий тракт НЕ трогать. Только аддитивно.

---

## 🎯 Цель

Детерминированное **угловое подавление** заградительной помехи (`barrage`) в домене
элементов решётки, ДО `Fft3DModel`. Доказать: помеха гаснет, цель под другим углом выживает.

## 📐 Математика (Doc/anti_barrage_math.md §3–4)

- Сырой `datacube: complex [nx, ny, K]` → `X = datacube.reshape(M, K)`, `M = nx·ny`.
- Ковариация `R = X @ X.conj().T / K` (`M×M`, эрмитова).
- EVD `R = Σ λ_m e_m e_mᴴ`; доминантные `n_jammers` собств. векторов = подпространство помехи.
- Ортопроектор `P⊥ = I − E_J E_Jᴴ` (E_J — стек топ-собств. векторов, ортонормир.).
- Очистка: `Y = P⊥ @ X`, вернуть `Y.reshape(nx, ny, K)`.
- Критерий barrage: `λ₁/σ_n² > η` (σ_n² ≈ медиана младших с.з.) + **occupancy** по дальности.

## 📋 Шаги

### Шаг 1 — подпакет `core/models/anti_barrage/`  ⬜
- `nuller.py`: класс `SubspaceNuller`:
  - `__init__(self, n_jammers: int = 1, oblique: bool = False, target_steering: np.ndarray | None = None)`.
  - `apply(self, datacube: np.ndarray) -> np.ndarray` — element-domain, `[nx,ny,K]→[nx,ny,K]`,
    НЕ мутирует вход (возвращает новый массив, dtype complex64/128 как на входе).
  - `report(self, datacube: np.ndarray) -> NullerReport` — `λ₁/σ_n²`, occupancy, флаг is_barrage.
  - Ортогональная проекция — дефолт (угол цели не нужен). Если `oblique=True` и задан
    `target_steering` — косая `E_{t|J}` (§4.2), сохраняет амплитуду цели.
  - numpy-эталон (`np.linalg.eigh` — R эрмитова). torch/GPU — НЕ здесь (phase2).
  - `NullerReport` — frozen dataclass (VO): `lambda_ratio: float`, `occupancy: float`, `is_barrage: bool`.
- `__init__.py`: реэкспорт `SubspaceNuller`, `NullerReport`.
- **Прочитать РЕАЛЬНЫЙ код** `core/generators/scene.py` (Synthesizer) — подтвердить форму/тип
  сырого куба `[nx,ny,K]` complex ДО правки; не гадать.

### Шаг 2 — демо `demo_nuller.py` (корень, как classify_demo.py)  ⬜
- Собрать сцену target+barrage под РАЗНЫМИ углами (SceneBuilder + specs из `core/config`).
- Прогнать `synth.build(scene)` → `SubspaceNuller().apply(raw)` → `Fft3DModel.process` до/после.
- Напечатать: подавление помехи (дБ по угловой энергии), выживание цели, `NullerReport`.
- Сохранить PNG до/после через существующий `AngularMapVisualizer` (не плодить визуализаторы).

### Шаг 3 — тест `tests/test_nuller.py` (TestRunner, БЕЗ pytest)  ⬜
- `NullerTests(TestRunner)` c `AssertionGroup`:
  - barrage-only: после `apply` остаточная энергия помехи падает ≥ ~20 дБ.
  - target+barrage разн. углы: угловой пик цели сохранён (в пределах допуска), помеха подавлена.
  - идемпотентность проектора: `P⊥² ≈ P⊥`, `P⊥ @ E_J ≈ 0`.
  - `report.is_barrage == True` для barrage-сцены, `False` для чистой target.
- Дописать `NullerTests` в `tests/all_test.py` SUITES (файл уже есть — не создавать заново).

## ✅ Definition of Done

- [ ] `python demo_nuller.py` — помеха подавлена, цель жива, отчёт напечатан, PNG до/после.
- [ ] `python tests/all_test.py` — всё зелено (Smoke + Graphics + Nuller).
- [ ] `mypy core/` 0 ошибок · `ruff check core/` чисто.
- [ ] Контроллер/существующий тракт НЕ изменён (проверить `git diff` — только новые файлы + all_test.py).
- [ ] Вход `apply` не мутируется (чистота).

## 🔮 Phase2 (НЕ сейчас)

Впайка нуллера в pipeline (опц. preprocessor в контроллере — с согласованием Alex),
torch/GPU-бэкенд, CFAR-детектор (§7), затем Доплер (§5).
