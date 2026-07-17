# 🧩 TASK — Body-Motion · P2 (splat цели + шум → входы обеих веток nx×ny×N)

> **Исполнитель:** Sonnet · **Ревью:** Кодо (Opus) · **Тип:** новый код (заполнение объёма сигналом).
> **Спека:** [`SPEC.md`](../specs/body_motion_3d_2026-07-15/SPEC.md) (§1,§7·S2).
> **Зависит от:** P1 (ProjectConfig, DataContext-шина, motion). **Статус:** ⏳ К РЕАЛИЗАЦИИ.
>
> 🚨 🚫 pytest (`TestRunner`) · 🚫 `.claude/worktrees/**` · существующее НЕ ломать · реюз, не плодить.

---

> 🧭 **Конвенция 3D-визуала (везде):** дальность (range) — по **горизонтали** (пол сцены, вдаль); kx (азимут) — вбок; ky (угол места) — вверх. Динамика: GIF + `--live` окно + `.html`.

## 🎯 Цель P2

По траектории из P1 **разместить движущуюся точечную цель** в комплексном объёме
`nx×ny×N` (апертура×апертура×дальность, N=1024…10000; nx×ny — произвольные, каждая ось = 2ⁿ,
недобор → zero-pad, частный случай 16×16) + тепловой шум. Объём — **сырой вход
фронтенда** (ещё БЕЗ FFT). Формируем **один и тот же формат входа для обеих веток** (ЛЧМ и
АМ едины по входу — `nx×ny×N`; различие в обработке — P5). Цель — **точечная** (этап 1;
протяжённая — позже, решение Alex).

## 🧭 Опорные решения (спека/патент)
- Вход обеих веток по **форме** — `nx×ny плоскостей × N по дальности` (§1).
- 🔴 **A4 — по ФИЗИКЕ входы РАЗНЫЕ** (патент): ЛЧМ = дечирпованный beat-тон (частота ∝ дальность,
  гл.3.1); АМ = отдельный амплитудно-модулированный зонд (гл.4-бис.2). Это **не** «один вход, разная
  обработка». **Решение для P2:** `VolumeBuilder.build(state, cfg, rng)` ветвится по `cfg.modulation`
  (`"lfm"`/`"am"`) — заложить параметр сейчас; на этапе точки (P2) обе ветки splat-ятся сигналом по
  дальности (тон в бине `R`), но **расхождение физики зондов явно помечено** и закрывается в P5
  (ЛЧМ: тон на всю Z; АМ: огибающая с редкими выбросами). НЕ молчаливое «вход один».
- Цель точечная → **splat** в апертуру (стиринг) + позиция по дальности из `Kinematics`.
- Движение по тактам: на каждый такт — свой объём (задел под slow-time/Доплер).

## 📦 Что создать / расширить

### 1. `core/generators/volume.py` — `VolumeBuilder` (Pure Fabrication) — 🔴 **РЕЮЗ генераторов (A9)**
`build(state, cfg, rng) -> np.ndarray[complex64]` формы `(nx, ny, N)`. **НЕ изобретать splat/шум** —
реюзить готовый слой генерации (вчера, numpy/Windows, 48+ тестов):
```python
wf   = WaveformFactory().create(Modulation.LFM if cfg.modulation=="lfm" else Modulation.AM)  # A4 решён
spec = WaveformSpec(fs, carrier_hz, n_samples=N, amplitude/snr_db, fdev_hz,
                    window=TimeWindow(...), meta={"kx":kx, "ky":ky, "nx":nx, "ny":ny, ...})   # из Kinematics
field = wf.render(NumpyBackend(), spec, rng)     # → SignalField.data (nx,ny,N): steering+шум уже внутри
vol   = field.data
```
- `WaveformFactory`/`Waveform.render`/`SignalField`/`NumpyBackend` — **готовое** (`core/generators/
  waveforms/`, `backends/`). `render_pipeline` уже делает steering-раскладку n×n + шум по `snr_db` — **не
  дублировать** (это и был ручной splat).
- **A4 решён реюзом:** ЛЧМ → `LfmWaveform` (getX центр.чирп), АМ → `AmWaveform` (огибающая) — физика
  зондов разная **из коробки**, `cfg.modulation` выбирает волну.
- 🔴 **A9-gap1 — позиция по дальности `R` сама (numpy `tau_s` игнорит!):** `render_pipeline` НЕ применяет
  `spec.tau_s` (его учитывает только GPU `HipBackend`). На Windows позицию по дальности задаём **окном**
  `TimeWindow(kind="short", t0=2R/c, dur=…)` (реюз `placement`) или сдвигом по fast-time — не полагаться
  на `tau_s` в numpy-пути.
- `vr` → фазовая прогрессия по тактам (задел Доплер) — через `meta`/фазу spec.
- Билинейная раскладка при дробных `(kx,ky)` — если нужна точнее целого бина (иначе `steering` как есть).

### 2. Интеграция с тактами — расширить `TactSequence` (создан в P1, A3)
На каждый такт: `MotionModel.propagate` → `Kinematics` → `VolumeBuilder.build` → положить
куб в шину `DataContext.publish("cube", vol)` (визуал — Observer).

### 3. Демо S2 — дополнить `demo_body_motion.py`
Прогон N тактов → объёмы `nx×ny×N` → **3D-энергия** (реюз `core/graphics` cube_scatter/
interactive) в `graphics/body_motion/p2_volume/`. Показать движение пятна цели по тактам.

## ♻️ Реюз (точные пути) — 🔴 **сначала генераторы (A9), потом низкоуровневое**
- **Генераторы (главное):** `core/generators/waveforms/factory.py` (`WaveformFactory`),
  `waveforms/{lfm,am,cw,...}.py` (`Waveform.render`), `waveforms/field.py` (`SignalField`,`Modulation`),
  `waveforms/base.py` (`WaveformSpec`), `backends/numpy_backend.py` (`NumpyBackend`). — раскладка+шум внутри.
- Окно/дальность: `core/generators/waveforms/placement.py` (`TimeWindow`).
- Низкоуровневое (только если не хватает генераторов): `grid.ArrayGrid.steering`, `sources.PointTarget`.
- Визуал: `core/graphics/` (`cube_scatter`, `sampling`), interactive plotly.

## ✅ Критерии приёмки
- Объём `(nx,ny,N)` complex64; цель даёт **один компактный пик** в апертуре на её `(kx,ky)`;
  позиция по дальности соответствует `R(state)`.
- По тактам пятно **двигается** согласно траектории P1 (проверка: argmax по апертуре/дальности
  трекается плавно).
- `vr`-фаза присутствует (заготовка Доплера), входные массивы не мутируются (чистота).
- Куб публикуется в шину; визуал-Observer срабатывает после publish.

## 🧪 Тесты (`TestRunner`)
`VolumeBuilderTests`: форма/dtype, позиция пика = `(kx,ky,R)`, SNR≈задан, движение по тактам.
Старые наборы целы, ruff/mypy 0.

## 🚫 Границы
Без помех (P3), без мульти-цели (P4), без FFT/квадратов (P5), без сокета (P6).
Цель — **точечная** (протяжённая позже).
