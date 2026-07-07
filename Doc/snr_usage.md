# 📈 SNR-эстиматор — инструкция по использованию

> Модуль `core/snr/` — оценка отношения сигнал/шум **двумя способами** (спектр CA-CFAR +
> статистика во времени) для сигнала точки с длительностью, позицией строба и шумом.
> Перенесён из `GPUWorkLib/PyPanelAntennas/SNR` (C++/Python) в чистый **numpy-эталон**.
>
> ⚠️ **torch НЕ нужен** — модуль на numpy+matplotlib. Работает на Windows и Linux одинаково.

---

## 1. Что это и зачем

- **Генератор** `PointSignalGenerator` — делает комплексный тон (CW) длительностью `duration_frac·N`,
  ставит его строб в позицию `left/right/center`, добавляет AWGN на всю длину. Амплитуда задаётся
  через входной SNR: `A = √(σ²·10^(SNR_in/10))`.
- **Спектральный оценщик** `SpectrumSnrEstimator` — FFT(окно) → |X|² → пик → CA/OS-CFAR по шуму → `SNR_fft`.
- **Статистический оценщик** `StatisticsSnrEstimator` — по «пустой» зоне вне строба оценивает σ², внутри —
  мощность; `SNR_stat = 10·log10((P_строб − σ²)/σ²)`.

**Ключевое отличие двух методов:**
| | processing gain | инвариантность |
|---|---|---|
| спектр | **есть** (+10·log10(N)) — растёт с длиной | зависит от длины/позиции строба |
| статистика | **нет** — даёт ~`SNR_in` | инвариантна к длине/позиции |

---

## 2. Установка

### 🪟 Windows (дома) — просто, без torch

Нужен **Python 3.11+** (подойдёт 3.12/3.13, torch тут не участвует, так что версия не критична).

```powershell
# 1. Клонировать/скопировать проект, зайти в папку
cd C:\path\to\Test_3FFT_model

# 2. Создать venv и активировать
python -m venv .venv
.venv\Scripts\activate

# 3. Поставить зависимости (только numpy + matplotlib!)
python -m pip install --upgrade pip
python -m pip install -e .
#   ↑ ставит numpy>=1.26 и matplotlib>=3.8 из pyproject.toml

# 4. Запустить демо (нарисует графики)
python demo_snr.py

# 5. Запустить тесты
python tests\test_snr.py
#   или все наборы:
python tests\all_test.py
```

Графики появятся в `out\figures\snr_curve.png`, `snr_boundary.png`, `snr_spectrum.png`.

> Если `pip install -e .` капризничает — можно напрямую: `python -m pip install "numpy>=1.26" "matplotlib>=3.8"`.

### 🐧 Linux (рабочая машина)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python demo_snr.py
.venv/bin/python tests/all_test.py
```

> torch-ROCm нужен ТОЛЬКО для обучаемой 3D-CNN (`train_cnn.py`) — к SNR отношения не имеет.

---

## 3. API — как пользоваться в коде

### 3.1. Сгенерировать сигнал точки

```python
import numpy as np
from core.snr import PointSignalGenerator

gen = PointSignalGenerator()
signal, support = gen.generate(
    n_samples=2048,        # полная длина по времени N
    freq_norm=0.15,        # нормированная частота тона f/fs ∈ (-0.5, 0.5)
    snr_in_db=20.0,        # входной SNR (per-sample в стробе), дБ
    duration_frac=0.3,     # доля длины: строб = 0.3·N (краевое условие!)
    position="left",       # "left" | "right" | "center"
    noise_power=1.0,       # σ² шума (НЕ 0 — иначе тон обнулится: A∝√σ²)
    rng=np.random.default_rng(42),   # для воспроизводимости
)
# signal  : np.complex64[2048] — тон в стробе + AWGN на всю длину
# support : slice — где именно лежит строб (ground-truth, нужен статистике)
```

> ⚠️ `noise_power=0` даёт `A=0` (тон исчезает), т.к. амплитуда привязана к SNR. Для «чистого» тона
> ставь высокий `snr_in_db` (напр. 40 дБ) с `noise_power=1`.

### 3.2. Оценить SNR — спектр (CA-CFAR)

```python
from core.snr import SpectrumSnrEstimator, SnrConfig

spec = SpectrumSnrEstimator(SnrConfig(window="hann", cfar_estimator="mean",
                                      guard_bins=3, ref_bins=8))
res = spec.estimate(signal)          # support не нужен спектру
print(res.snr_db, res.k_peak, res.peak, res.noise)
```

### 3.3. Оценить SNR — статистика (время)

```python
from core.snr import StatisticsSnrEstimator

stat = StatisticsSnrEstimator()
res = stat.estimate(signal, support)   # support ОБЯЗАТЕЛЕН (иначе ValueError)
print(res.snr_db, res.p_signal, res.noise_var)
```

> Статистике нужна «пустая» зона вне строба → `duration_frac < 1.0`. При `frac=1` оценка вырождается.

### 3.4. Конфиг спектра (`SnrConfig`, Value Object)

| поле | по умолчанию | смысл |
|------|--------------|-------|
| `target_n_fft` | 2048 | целевой размер FFT (0 → авто 2048) |
| `step_samples` | 0 | децимация по времени (0 → авто) |
| `guard_bins` | 3 | защитные ячейки вокруг пика |
| `ref_bins` | 8 | опорные ячейки для оценки шума |
| `window` | `"hann"` | `rect`/`hann`/`hamming`/`blackman` |
| `cfar_estimator` | `"mean"` | `mean` (CA-CFAR) / `median` (OS-CFAR) |

---

## 4. Демо и графики

`python demo_snr.py` рисует 3 графика в `out/figures/`:

1. **`snr_curve.png`** — измеренный SNR (спектр + статистика) vs `SNR_in`. Спектр насыщается
   (короткий строб), статистика идёт ровно `y=SNR_in` (нет processing gain).
2. **`snr_boundary.png`** — **краевые условия**: SNR vs `duration_frac ∈ {0.3,0.2,0.1,0.05}`.
   Спектр 3 линиями (left/center/right — позиция влияет через тапер Hann), статистика — ровная.
3. **`snr_spectrum.png`** — спектр |X|² одной реализации: пик + CFAR ref-окна.

---

## 5. Тесты

```bash
python tests/test_snr.py     # только SNR (7 проверок)
python tests/all_test.py     # все наборы проекта
```
🚫 pytest в проекте запрещён — используется `common.runner.TestRunner`.

---

## 6. Расширение (phase2, не сделано)

Медиана по антеннам (`estimate_batch`), `BranchSelector` (Low/Mid/High по порогам 15/30 дБ),
калибровка порогов, dechirp для полного LFM, torch/GPU-бэкенд FFT. См. `MemoryBank/tasks/TASK_snr_phase1.md`.

---

*Источник порта: `/home/alex/C++/GPUWorkLib/PyPanelAntennas/SNR/` (cfar_estimator.py, lfm_signal_generator.py).*
