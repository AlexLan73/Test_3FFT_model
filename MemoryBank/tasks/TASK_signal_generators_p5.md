# 🧩 TASK — Генераторы сигналов · P5 (помехи: патент + промышленные)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** новый код (время-домен помехи), реюз пайплайна.
> **Спека:** [`specs/signal_generators_2026-07-13.md`](../specs/signal_generators_2026-07-13.md) (§3, §3.1, §5)
> + [`specs/industrial_interference_classification.md`](../specs/industrial_interference_classification.md) (§4-6).
> **Зависит от:** P0+P1+P2+P4 (`SignalField`, `Waveform`, `render_pipeline`, `WaveformFactory`, `reference`, `mseq`).
> **Статус:** ✅ ПРИНЯТО (ревью Кодо 2026-07-14). 6 помех: python3.13 47 ok/0 fail, .venv 39/8skip.
> Barrage rank-1 (J4) ✅, IMP_ARC kurtosis=260/разреж.1.1% (J5/J6) ✅, DRFM 4 пика на τ_i, SMSP размаз
> после дечирпа (J2), INT_CW пик 3.48МГц, VFD гребёнка n·f_sw. J1(snr_db)/J3(Найквист)/J7 в коде верны.
> ruff/mypy чисто, старые целы. Визуал `p5_jammers/` — все 6 сигнатур. (ревью-правки Alex J1-J7 — отработали)
>
> 🚨 🚫 pytest (только `TestRunner`+`AssertionGroup`+`SkipTest`) · 🚫 `.claude/worktrees/**` · существующее НЕ ломать.

---

## 🎯 Цель P5

Время-доменные **помехи** на n×n → `SignalField`, единым пайплайном (реюз `render_pipeline`).
Приоритетный набор: **3 патентные** (§1.2) + **3 промышленные** (industrial §5 приоритет). Остальной
зоопарк (CLUT_*, MOT_BRUSH, CORONA_HV, RF_WIFI) — позже. ЧМ-радиолюбитель уже есть (`FmInterferenceWaveform`, P4).

## 🏗️ Дизайн (единый подход)

Каждая помеха = **`Waveform`-подкласс** в `core/generators/waveforms/jammers_rf.py`, отдаёт `SignalField`
через **реюз `_pipeline.render_pipeline`** (окно/раскладка n×n/шум) — как cw/lfm/am/phase_code. Отличие —
только **шаг 1** (сырой 1D-сигнал помехи). Угол `(kx,ky)` и мощность — из `spec.meta`/`spec.amplitude`
(геометрию задаёт SceneModeler позже, R3). Новые значения `Modulation` для тегов (self-documenting).

> Куб-уровневые `DrfmComb`/`BarrageJammer`/`HamEmitter` (`core/generators/jammers.py`) — **НЕ трогаем**
> (другой домен, после-дечирп куб). Берём их только как **конвенцию параметров** (kx,ky,lead,spacing,count,power).

## 📦 Что создать — `core/generators/waveforms/jammers_rf.py`

Добавить в `Modulation` (field.py): `BARRAGE, SMSP, DRFM_REPEATER, INDUSTRIAL_CW, IMPULSIVE_ARC, VFD_HARMONIC`.

### Патентные (§1.2 модель угрозы)
1. **`BarrageRfJammer`** — заградительная: широкополосный шум с одного угла.
   `s(t) = √power · white_complex_noise(n)` (когерентный по элементам через steering).
   Спектр — равномерно поднятый пол. `Modulation.BARRAGE`.
2. **`SmspJammer`** — размытие спектра (Smeared Spectrum): DRFM пересобирает ЛЧМ сегментами с
   **увеличенной** скоростью чирпа → спектр размазывается. Модель: `K` подчирпов, каждый свипует
   всю полосу за `1/K` длины (`μ_smsp = K·μ`), состыкованы. `Modulation.SMSP`.
3. **`DrfmRepeaterJammer`** — гребёнка ложных целей: задержанные копии опорного ЛЧМ.
   `s(t) = Σ_{i=0}^{count-1} a·decay^i · lfm_ref(t − τ_i)`, `τ_i = lead + i·spacing` (секунды).
   Реюз `reference.getX_numpy` для опорного ЛЧМ (та же формула зонда). `Modulation.DRFM_REPEATER`.
   ⚠️ Задержка — сдвиг по отсчётам `round(τ_i·fs)`, копии за пределом окна отбрасываются.

### Промышленные (industrial §5-6, топ-3 по приоритету)
4. **`IndustrialCwJammer`** (`INT_CW`, 🔴1) — CW чужого радара: `A·exp(j2π·f_int·t)`. Острый пик,
   имитатор цели. `f_int` в `spec.meta`. (можно реюз `reference.cw_numpy`.) `Modulation.INDUSTRIAL_CW`.
5. **`ImpulsiveArcJammer`** (`IMP_ARC`, 🔴2) — сварочная дуга/разряд: пуассоновский поток импульсов
   с тяжёлыми хвостами. `s(t) = Σ_k A_k·h(t−t_k)`, `h(u)=exp(−u/τ_decay)·[u≥0]` — затухающий импульс
   в `t_k` (🐞 **J5:** это свёртка `δ*h`, НЕ буквальное `δ·exp` — иначе одиночный отсчёт без хвоста).
   `t_k` — пуассон (интенсивность λ). 🐞 **J6:** сигнал комплексный → `A_k = r_k·exp(jφ_k)`,
   `r_k`~α-stable (`scipy.stats.levy_stable`, α≈1.4; фолбэк — Стьюдент/смесь гаусс), `φ_k`~U(0,2π).
   Высокий эксцесс (kurtosis≫3), разреженность во времени. `Modulation.IMPULSIVE_ARC`.
6. **`VfdHarmonicJammer`** (`HAR_VFD`, 🔴3) — гармоники VFD/IGBT: `Σ_{n=1}^{N} A_n·cos(2π·n·f_sw·t)`
   + широкополосная компонента. `f_sw` (2-16 кГц), спад `A_n` с `n`. Линейка пиков в спектре.
   `Modulation.VFD_HARMONIC`.

### Калибровка мощности (R5) — J1: реюз готового пути, не плодить второй
- Уровень помехи задаём через **существующий `spec.snr_db`** (здесь это **JNR** — та же математика
  `A=√(σ²·10^(ratio/10))`, тот же эталон `σ²=NOISE_POWER`) → зовём **готовый `amplitude_for_snr(spec)`**.
  🐞 **J1:** НЕ вводить отдельный `jnr_db` в `meta` — это второй путь калибровки с риском иного `σ²`
  и рассогласования с `add_noise(…, NOISE_POWER)`. Одна шкала дБ, один эталон мощности шума.
- Абсолютная мощность (без привязки к шуму) — через `spec.amplitude` (при `snr_db=None`).
- **R6:** вся случайность (шум, пуассон, α-stable, фазы) — через переданный `rng`.

### Регистрация — `WaveformFactory` (реюз, §4.2): 6 новых `Modulation` → классы.

## ♻️ Реюз (точные пути)
- Пайплайн/формулы: `core/generators/waveforms/{_pipeline,reference,mseq}.py`.
- Конвенции параметров помех: `core/generators/jammers.py` (kx/ky/lead/spacing/count/power), `core/config/scene_config.py` (spec-VO).
- Формулы/приоритеты: `MemoryBank/specs/industrial_interference_classification.md` §6 (мат.модели), §5 (приоритет).
- Шум: `backends/numpy_backend.py::add_noise`.

## ✅ Definition of Done
- Тесты (`tests/test_generators.py`, `TestRunner`, numpy — GPU не нужен):
  - **Barrage:** спектр широкополосный (≈равномерный пол), не узкий пик; энергия под нужным углом (steering).
  - **SMSP (J2 — тест по ДЕЧИРПУ, не по сырому спектру):** сырой чирп и SMSP занимают ТУ ЖЕ полосу →
    ширина сырого спектра их НЕ различает. Признак SMSP виден **после дечирпа опорным ЛЧМ**: matched ЛЧМ →
    один острый пик, SMSP → размазанный/множественные пики (реюз `reference.getX_numpy` + `dechirp_numpy`).
  - **DRFM-repeater:** кросс-корреляция с опорным ЛЧМ даёт **`count` пиков** на `τ_i` (гребёнка задержек).
  - **INT_CW:** острый пик спектра на `f_int`.
  - **IMP_ARC:** высокий эксцесс (kurtosis ≫ 3) + разреженность (малая доля отсчётов выше порога).
  - **VFD:** пики спектра на `n·f_sw` (n=1..N).
  - **Форма/тип/детерминизм:** `SignalField` (16,16,n) complex64, один seed → идентично (R6).
- `.venv/bin/python tests/all_test.py` — зелено (существующие P0/P1/P2/P4 не сломаны).
- `ruff check core/generators tests` + `mypy core/generators` — чисто.

## 🖼️ Визуал-подтверждение (обязательно, §9-конвенция)
Каталог `graphics/signal_generators/p5_jammers/` (`FigureWriter`), демо в `demo_generators.py`:
- `jammers_spectra.png` — 6 сабплотов: спектр каждой помехи (barrage-пол, smsp-размаз, drfm-гребёнка,
  int_cw-пик, imp_arc-импульсы во времени, vfd-линейка гармоник).
- `scene_target_plus_jammers.png` — цель (ЛЧМ) + 2-3 помехи в одном `SignalField` (сумма вкладов) —
  наглядно «полезный сигнал на фоне зоопарка» (модель угрозы патента §1.2).

## ⚠️ Подводные камни
- 🐞 **J3 — частоты помех в СИМУЛИРУЕМОЙ baseband-полосе:** генерация на `fs=12МГц`, `f0=2МГц` (§5.1).
  Все `f_int`/`f_sw`/`n·f_sw` обязаны быть `< fs/2` (иначе алиасинг). VFD-гребёнка при `f_sw=2–16кГц` ляжет
  у DC (далеко от РЛ-линии `f0`) — решить: маппить гребёнку в полосу (около `f0`) или оставить НЧ.
  Это полосовая модель приёмника, не «настоящий РЧ» — выбор задокументировать.
- 🐞 **J4 — barrage когерентный (из угла), не диффузный:** `render_pipeline` множит ОДИН 1D-шум на steering →
  массив rank-1 (когерентная волна из `(kx,ky)`) = направленный jammer. Пространственно-белый «поднятый пол»
  (независимый шум на элемент) через `render_pipeline` НЕ получить; белый тепловой пол уже даёт `add_noise`.
  Строим **направленный** barrage; диффузный — отдельная модель (не сейчас).
- 🐞 **J7 — тепловой шум в сцене добавлять ОДИН раз:** `render_pipeline` льёт `add_noise` при `snr_db!=None`.
  В сумме `цель+помехи` (visual `scene_target_plus_jammers`) это добавит шум на КАЖДЫЙ источник (двойной счёт).
  В сцене: источники генерить без пер-источникового шума (уровень — через `amplitude`), тепловой пол
  добавить один раз на суммарное поле.
- **R6:** ТОЛЬКО переданный `rng` (много случайности: шум/пуассон/α-stable/фазы) — воспроизводимость датасета.
- **DRFM задержки:** `round(τ·fs)` по отсчётам; копии за окном отбрасываются; опорный ЛЧМ = `getX` (как зонд).
- **α-stable:** `scipy.stats.levy_stable` может быть медленным — генерить пачкой; при отсутствии/тормозах — фолбэк (смесь гауссиан / Стьюдент), задокументировать.
- Не мутировать входы; помехи — новые файлы, реюз `render_pipeline` (не дублировать пайплайн).
- НЕ трогать куб-уровневые `jammers.py`/`sources.py` и P0/P1/P2/P4.

## 🚫 Вне P5
Остальной зоопарк (клаттер CLUT_*, MOT_BRUSH, CORONA_HV, RF_WIFI — по мере надобности), движение/такты
(P6), сцена-моделлер полноценный (R3, позже), порт в C++ (P7).
