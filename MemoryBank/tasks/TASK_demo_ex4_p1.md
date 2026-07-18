# 📋 TASK — demo ex4: летящие цель и помехи + барьер + анимация (p1)

> Создан: 2026-07-18 · Схема: **Кодо ТЗ → Sonnet код → глубокое ревью Кодо → приёмка → пуш**
> Спека-канон: [`specs/demo_ex4_flight_2026-07-18.md`](../specs/demo_ex4_flight_2026-07-18.md)
> (решения Alex §1, реюз §2, конвейер §3, тесты §4, ревью R1-R6).
> Требование Alex: **НЕ сочинять код заново — брать готовое** (карта ниже).

## Карта реюза (точные адреса — импортируй, не переписывай)

| Нужно | Взять | Адрес |
|---|---|---|
| Случайный старт/скорость | `_random_initial_state(rng)` — ОБРАЗЕЦ (приватный, скопировать паттерн 10 строк с атрибуцией) | `demo_body_motion.py:57` |
| Случайные модели | образцы `_random_maneuver`/`_random_markov` `:68/:76` + выбор из 5: `rng.choice` | `core/motion/models.py` (все классы готовы) |
| Тёмный стиль | `_BG="#0d1117"`, `_style_axes` `:110`, rcParams-блок `:164` — тот же приём | `demo_body_motion.py` |
| GIF | `FuncAnimation`+`PillowWriter(fps=5)` | `demo_body_motion.py:259-261` |
| plotly HTML | блок `:223-242` (`template="plotly_dark"`, `HtmlWriter(OUT).write(fig,"...html")`) | там же + `core/graphics/interactive` |
| Проекция state→(kx,ky,r) | `Kinematics(cfg).project(state, dt)` | `core/motion/kinematics.py:34` |
| Эхо цели (S1) | `build_pulse_echo_volume(...)` | `core/generators/waveforms` (реэкспорт) |
| Гребёнка за носителем | `build_drfm_comb_volume(p, kx, ky, rng)` — угол = позиция НОСИТЕЛЯ такта | `demo/ex3_am_barrage/example.py` |
| Barrage | `build_jammer_volume(p,"barrage",...)` | там же |
| Полоса→null→скан | `_run_pipeline`-логика: `band_angle`/`null_angle` + `coarse_burst_points`/`detect_objects` | ex3/ex2 example.py |
| Признаки §4.11 в токенах | `VolumeTokenizer` уже возвращает — из `fine_scan_roi`/токенов | ex2 |
| Трек + «летит» | `NearestNeighborTracker(...)`, `Track.is_moving` | `core/models/tracking/` |
| 3D-кадр | `ScenePointsVisualizer.render(..., ax=)` | `core/graphics/scene_points.py` |
| Шум на такт | `add_noise_volume(vol, snr, rng)` | ex2 |

## Файлы задачи

1. `demo/ex4_flight/__init__.py` + `example.py` — `Ex4Params` (композиция Ex3Params/Ex2Params;
   nx/ny — параметр, дефолт 64×64×4096, НЕ хардкодить: понедельник 512×256), `Ex4Flight(DemoRunner)`.
2. Движущиеся сущности: цель (эхо S1), носитель гребёнки (те же модели), barrage-угол
   (дрейф ≤0.5 бина/такт). Модель каждому — `rng.choice` из 5, старт/скорость случайно
   (клиппинг: не вылететь за поле/ось за 30 тактов).
3. Компоновка кадра (спека §1.6, ТЁМНАЯ): слева 3D · справа поле nx×ny (следы K=8 + №)
   · внизу ряд срезов по ВСЕМ трекам (16-окно вокруг позиции, № на срезе) + строка параметров
   (№, окно, kx, ky, PR, Hoyer, MainFrac, LobeRatio, MaxMean, Energy, «летит»=Track.is_moving).
4. Выходы: `flight_trail.gif`/`flight_clean.gif` (из ОДНОЙ истории), `flight_3d.html`
   (plotly), `trajectory.png`, `last_frame.png` → `demo/graphics/ex4_flight/`.
5. `demo/tests/test_ex4.py` (спека §4: 64×64×4096, 6 тактов, детерминизм/движение/found/
   трек/файлы>0; plotly и Pillow — под `SkipTest`) + регистрация в `all_demo_test.py`/`run_all.py`.

## Приёмка Кодо (после Sonnet)

- [ ] `git diff` прочитан: реюз по карте, БЕЗ переписанных формул/моделей/стилей.
- [ ] Тесты demo + бэкенд — гоняю сама; время прогона замерено.
- [ ] GIF оба открыты глазами: тёмный стиль, хвост K=8 vs без, объекты с №.
- [ ] HTML вращается (спот-чек), поле справа = параметр.
- [ ] Срезы по всем объектам, «летит» из трека.
