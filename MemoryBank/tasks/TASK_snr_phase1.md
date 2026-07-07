# 🧩 TASK — SNR estimator (статистика + спектр) · phase1

> **Тип:** порт рабочего Python-кода из GPUWorkLib в radar3d (НЕ исследование — рабочий код + графики).
> **Статус:** ✅✅ РЕАЛИЗОВАНО (Sonnet ядро + Кодо demo/тесты/фикс, 2026-07-07). all_test 28 ok (Snr 7),
> mypy/ruff чисто, 3 графика. Кодо на прогоне поймал 2 бага: frac=1 вырождает статистику (свип→frac=0.5),
> noise_power=0 обнуляет тон (A∝√σ²) → тесты переписаны. Инструкция: `Doc/snr_usage.md` (+Windows, torch-free).
> **Критично:** 🚫 pytest (только `common.runner.TestRunner`) · 🚫 писать в `.claude/worktrees/**`.
> **Неинвазивно:** пространственные генераторы (`core/generators/sources.py`, куб) НЕ трогать.

---

## 🎯 Цель

Перенести из `/home/alex/C++/GPUWorkLib/PyPanelAntennas/SNR/` рабочую Python-модель SNR-эстиматора
в наш проект **по аналогии, на наше ядро** (numpy-эталон, ООП/SOLID/GoF). Результат — **рабочий код +
графики**: генерация сигнала точки во времени с длительностью и шумом + оценка SNR двумя способами
(статистика во времени + спектр CA-CFAR) + краевые условия по длительности.

## 📚 Источник (референс — читать, портировать логику)

- `.../SNR/lfm_signal_generator.py` — `make_cw` (тон), `make_awgn`, `make_cw_with_snr` (A=σ·10^(SNR/20)).
- `.../SNR/cfar_estimator.py` — `estimate_snr_one_antenna` (decimation→window→zero-pad pow2→FFT→|X|²→
  argmax→CA/OS-CFAR по guard/ref→`10log10(peak/noise)`), `compute_pipeline_sizes`, `make_window`.
- `.../SNR/snr_estimator_model.py` — драйвер экспериментов/графиков (взять ИДЕИ графиков, не копировать 6 эксп.).

## 🚨 Обязательное требование Alex — краевые условия

Полезный сигнал занимает **долю** полной длины `T`: **0.3, 0.2, 0.1 и меньше**, и располагается у
**края** (слева/справа) и по центру. Т.е. тон — «строб» длительностью `duration_frac·T`, помещённый по
смещению; остаток окна — нули; **шум добавляется на всю длину `T`**. Цель — увидеть, как SNR-оценка
(и статистика, и спектр) деградирует, когда полезный сигнал короткий и прижат к границе.

## 🏗️ Дизайн (на ревью Opus — критиковать/утвердить)

Наш тон = формула `_SteeredTone._tone` (`amp·exp(j(2π·freq·k+phase))`) — **совпадает** с `make_cw` →
переиспользуем, генератор точки НЕ копируем 1-в-1, а делаем временну́ю обёртку.

Предлагаемое размещение (аддитивно, отдельный когезивный модуль):
- `core/snr/config.py` — `SnrConfig` (VO, frozen): `target_n_fft, step_samples, guard_bins, ref_bins,
  search_full_spectrum, window, cfar_estimator` (аналог `CfarConfig`). + `next_power_of_2`,
  `compute_pipeline_sizes` (helpers).
- `core/snr/signal.py` — `PointSignalGenerator` (генератор сигнала точки во времени):
  `generate(n_samples, freq_norm, snr_in_db, duration_frac, position, noise_power, rng) -> np.complex64[n]`.
  Строб-тон длительностью `round(duration_frac·n_samples)` по позиции `left|right|center` + AWGN на всю длину.
  Амплитуда из SNR: `A=√(noise_power·10^(SNR/10))`. Позиция — enum/Literal.
- `core/snr/estimator.py`:
  - `SnrEstimator(Protocol)`: `estimate(signal, support=None) -> SnrResult`. Спектр **игнорирует** support;
    статистика **требует** support (raise `ValueError` если None). Один Protocol → один драйвер графиков.
  - `SpectrumSnrEstimator(SnrConfig)` — порт `estimate_snr_one_antenna` (окна rect/hann/hamming/blackman,
    CA-CFAR mean / OS-CFAR median, wraparound ref-окна). ⚠️ это 1D SNR-ratio по временно́му ряду —
    в докстроке явно отличить от `core/models/anti_barrage/CaCfarDetector` (детектор по оси дальности куба).
  - `StatisticsSnrEstimator` — **статистика во времени** (НОВОЕ, в референсе нет). Формулы (Opus, дать явно):
    ```
    σ̂²        = mean(|x_k|²  по «пустым» отсчётам ВНЕ строба; при center — оба края объединить)
    P̂_total   = mean(|x_k|²  по стробу support)
    P̂_signal  = max(P̂_total − σ̂², ε)          # ε=1e-30; вычитание σ² обязательно!
    SNR_stat_dB = 10·log10(P̂_signal / σ̂²)
    ```
    Наивное `P_signal = mean(|x|² по стробу)` — ЗАПРЕЩЕНО (сместит +σ², враньё на низком SNR).
    Следствие: `E[P̂_signal]=A²` **не зависит от frac** → `SNR_stat ≈ SNR_in`, processing gain отсутствует
    (в этом смысловое отличие от спектра). Редукции σ̂²/P̂ — в **float64** (`.astype(np.float64)` до mean).
    Статистике `SnrConfig` НЕ нужен (ISP) — конфиг только спектральный.
  - `SnrResult` — frozen dataclass (VO): общие `snr_db: float`, `method: str`; метод-специфичные —
    `k_peak/peak/noise` (спектр), `p_signal/noise_var` (статистика) как `float | None = None`.
- `core/snr/__init__.py` — реэкспорт публичного API.

> ⚠️ matplotlib — только в demo (не в библиотечном коде `core/snr/`). Библиотека НЕ печатает/НЕ рисует.

## 📋 Шаги

### Шаг 1 — `core/snr/config.py` + helpers  ⬜
`SnrConfig` (frozen VO) + `next_power_of_2` + `compute_pipeline_sizes` (порт 1-в-1, с type hints).

### Шаг 2 — `core/snr/signal.py` — генератор  ⬜
`PointSignalGenerator` со стробом (duration_frac + position) + AWGN на всю длину. Вход не мутирует,
`np.complex64`, явный `rng`. Формулу тона `A·exp(j(2πf·k+φ))` переписать локально (как `make_cw`),
НЕ тянуть приватный `_SteeredTone._tone` (он завязан на RangeConfig/range_bin).
- SNR_in — это per-sample in-strobe отношение `A²/σ²` → `A=√(noise_power·10^(SNR/10))`.
- **Возвращать `(signal, support)`** где support — slice/индексы строба (ground-truth для статистики,
  НЕ детектировать энергетически — на низком SNR ненадёжно, испортит краевой эксперимент).
- Позиция `left|right|center` (Literal/enum) → смещение строба длины `L=round(frac·n_samples)`.

### Шаг 3 — `core/snr/estimator.py` — два оценщика  ⬜
`SpectrumSnrEstimator` (порт CA/OS-CFAR) + `StatisticsSnrEstimator` (время) + `SnrEstimator(Protocol)` + VO.

### Шаг 4 — `demo_snr.py` (корень)  ⬜  ← ГРАФИКИ (Alex хочет видеть)
Общие параметры демо/тестов: **N=2048 → step=1** (чистая физика строба, без децимации-конфаунда),
`freq_norm=0.15` (вдали от DC и n_fft/2, wraparound ref-окна безопасен), `window=Hann`.
- (а) кривая: измеренный SNR (оба метода) vs SNR_in (свип, frac=1), + **строб-поправленная теория**:
  `SNR_fft ≈ SNR_in + 10·log10(n_actual) + 20·log10(frac)` (при frac=1 → базовая); статистика ≈ линия y=x.
- (б) **краевые условия** (SNR_in≈**5 дБ**, frac ∈ {0.3,0.2,0.1,0.05}):
  - спектр — **3 линии** {left,right,center} (позиция влияет ЧЕРЕЗ тапер Hann: центр w≈1, край Hann(0)=0
    → строб гаснет; с rect эффекта позиции НЕ будет — потому Hann);
  - статистика — **одна** линия + error-band (std по trials); позицию НЕ рисуем (инвариант сдвига);
  - ось Y широкая (left/right при frac=0.05 сядут к H0). Ожидаемо: спектр ↓ с frac и зависит от позиции;
    статистика ~плоская по среднему, растёт разброс на низком SNR.
- (в) спектр одной реализации: |X|² (дБ) с отметкой пика `k_peak` и ref-окна CFAR (guard/ref).
- Сохранить PNG в `out/figures/snr_*.png`. Печать сводки в консоль.

### Шаг 5 — тест `tests/test_snr.py` (TestRunner, БЕЗ pytest)  ⬜
Все тесты — фикс `np.random.default_rng(seed)`. Допуски (Opus):
- генератор: структурные проверки на **noise_power=0** (или по возвращённому `support`) — иначе шум залит
  на всю длину и «число ненулевых» ≠ frac·N. Проверить `len(signal)`, dtype complex64, `len(support)≈round(frac·N)`,
  позицию строба у края (left→начало, right→конец).
- монотонность: SNR_in ∈ {0,20,40} дБ (широко разнесены), усреднить по ≥20 seed — растёт для обоих методов.
- спектр на чистом тоне (noise_power=0, frac=1): `k_peak = round(freq_norm·n_fft) ±1`;
  `SNR_fft ≈ SNR_in + 10log10(n_actual)` допуск ±3 дБ (Hann loss 1.76).
- статистика на высоком SNR (30 дБ, frac=1): `|SNR_stat − SNR_in| < 1.0` дБ.
- H0 (чистый шум, seed фикс): статистика — `P̂_signal` клампится ε → `SNR_stat < 3 дБ` и конечна;
  спектр — `SNR_fft_H0 < 0.5·SNR_fft(SNR_in=30)` (или явный численный бордюр под тестовый N).
- Дописать `SnrTests` в `tests/all_test.py` SUITES (файл есть — не создавать заново).

## ✅ Definition of Done

- [ ] `python demo_snr.py` — 3 графика в `out/figures/snr_*.png`, консоль-сводка.
- [ ] Краевые условия видны: **спектр** — монотонное падение по frac (0.3→0.05) + расхождение left/right/center
  (через тапер Hann); **статистика** — среднее ~плоское, но растёт std по trials на низком SNR.
- [ ] `python tests/all_test.py` — всё зелено (+ SnrTests).
- [ ] `mypy core/` 0 ошибок · `ruff check core/ tests/` чисто.
- [ ] Пространственный тракт/куб не тронут (`git diff` — только новые файлы + all_test.py).
- [ ] Генераторы не мутируют вход; matplotlib только в demo.

## 🔮 Phase2 (НЕ сейчас)

`estimate_batch` / медиана по антеннам (multi-antenna, exp2/4/5 референса), torch/GPU-бэкенд FFT,
`BranchSelector` (Low/Mid/High по порогам 15/30 дБ + гистерезис), калибровка порогов (эксп.5),
dechirp для полного LFM, авто-детекция строба (энергетический детектор).
