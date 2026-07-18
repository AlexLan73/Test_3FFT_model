# anti-barrage phase2: pipeline + diagonal loading (2026-07-18)

> Схема: Кодо ТЗ → Sonnet код (упал на обрыве) → Кодо дочитала/дописала тесты → ревью Кодо.

## Сделано
- **`AntiBarragePipeline`** (`core/models/anti_barrage/pipeline.py`, Facade): единый тракт
  `SubspaceNuller.apply` (подавить заград по углу, сырой куб) → `RadarModel.process` (Fft3DModel,
  согласующее звено домена) → `CaCfarDetector.detect` (детект по дальности, SpectralCube). DI трёх
  готовых компонентов. Проверено Кодо: `process` ≡ ручной цепочке; цель детектируется под заградом;
  вход не мутируется.
- **diagonal loading** в `SubspaceNuller` (параметр `loading`, `R' = R + loading·(tr(R)/M)·I`).

## ⚠️ Находка (математика, зафиксирована тестом + докстрингом)
**diagonal loading в subspace-nuller НЕ влияет на подавление (`apply`).** `R+λI` сдвигает
собственные ЗНАЧЕНИЯ, но НЕ меняет собственные ВЕКТОРЫ. И ортогональная (§4.1), и косая (§4.2)
проекции строятся по собственным ВЕКТОРАМ подпространства помехи (`e_jam`) → **`apply` инвариантен
к loading**. Loading влияет ТОЛЬКО на собственные значения → `report.lambda_ratio` и детектор
`is_barrage` (стабилизация оценки числа источников/порога при малой выборке K).

Мой первый тест (ожидал, что loading меняет `apply`) корректно это вскрыл. Для *робастного
подавления* при малой K нужен **MVDR-подход с обращением `R⁻¹`** (там loading критичен) — это
отдельная задача, а не subspace-проекция. Тест `test_loading_affects_report_not_apply` документирует:
loading меняет `report.lambda_ratio`, но `apply` инвариантен.

## Тесты
`tests/test_anti_barrage_pipeline.py`: `DiagonalLoadingTests` (3: cond падает от loading на R;
loading=0 обратная совместимость; loading влияет на report не на apply) + `AntiBarragePipelineTests`
(3: Facade≡ручная цепочка; цель под заградом; немутируемость). Регистр в `all_test.py`.
nuller/cfar обратная совместимость целы (loading дефолт 0). Весь бэкенд **35 наборов, 215 ok, 0 fail**.

## На будущее
Робастный nuller (MVDR + diagonal loading, обращение R) · ROI-гейт детекции · угловая кластеризация.
