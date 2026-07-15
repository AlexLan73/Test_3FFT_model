"""VolumeBuilder -- заполнение входного объёма (nx, ny, N) под цель (Pure Fabrication, P2).

Куб -- **сырой вход фронтенда**, ещё БЕЗ FFT (SPEC §1/§2): один и тот же формат
`nx x ny x N` для обеих веток (ЛЧМ/АМ), физика зондов внутри РАЗНАЯ (A4).

🔴 A4 (TASK P2, решено реюзом): `cfg.modulation` выбирает `Waveform` через
`WaveformFactory` -- ЛЧМ → `LfmWaveform` (центрированный чирп, `reference.getX_numpy`),
АМ → `AmWaveform` (огибающая). `render_pipeline` (`core/generators/waveforms/_pipeline.py`)
уже делает steering-раскладку n×n по `(kx, ky)` из `spec.meta` и добавляет шум,
калиброванный по `spec.snr_db` -- **не дублируем** этот splat/шум вручную здесь.

🔴 A9-gap1 (TASK P2): numpy-бэкенд игнорирует `WaveformSpec.tau_s` (только
`HipBackend` его учитывает). Позицию цели по дальности задаём **коротким временны́м
окном** `TimeWindow(kind="short", t0=2R/c, dur=...)` (реюз `waveforms.placement`),
а не через `tau_s` (`tau_s` всё равно прокидываем в spec -- для GPU-бэкенда позже).

Билинейная раскладка по дробным `(kx, ky)` отдельно НЕ реализуется: `ArrayGrid.
steering(kx, ky)` и так честный непрерывный фазовый вектор (не требует раскладки по
соседним целочисленным бинам) -- то есть эта задача уже решена реюзом, дублировать
нечего (см. TASK п.1, "билинейная раскладка ... иначе steering как есть").
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from ..config import ProjectConfig
from ..data_context import DataContext
from ..motion import Kinematics, KinematicsSample, TargetState
from .backends import NumpyBackend
from .tact_sequence import MultiTact, Tact
from .waveforms import Modulation, TimeWindow, WaveformFactory, WaveformSpec
from .waveforms._pipeline import NOISE_POWER  # noqa: SLF001 -- тот же приём, что hip_backend.py

C_LIGHT = 299_792_458.0  # м/с (совпадает с core.motion.kinematics.C_LIGHT)

CUBE_CHANNEL = "cube"    # канал шины (SPEC §4): публикуется сырой объём (nx,ny,N)

_MODULATION_BY_NAME = {"lfm": Modulation.LFM, "am": Modulation.AM}


@dataclass(frozen=True)
class VolumeBuilder:
    """Строит входной объём `(nx, ny, N)` под текущее состояние цели.

    Параметры -- рантайм-настройки заполнения (не геометрия решётки: та -- в
    `ProjectConfig.array`/`cfg.wave`, не дублируем).

    n_samples  -- N по фаст-тайм (SPEC §1: 1024..10000). Дефолт 1024 -- параметр
                  ОТДЕЛЬНЫЙ от `cfg.wave.n_samples` (у `WaveTimeConfig` дефолт 8192,
                  что для демо-прогонов по многу тактов избыточно медленно) -- сделан
                  явным полем `VolumeBuilder`, а не привязан к `cfg.wave.n_samples`.
    snr_db     -- калибровка амплитуды цели относительно шумового пола (R5-мат.,
                  реюз `amplitude_for_snr`/`backend.add_noise` внутри render_pipeline).
    pulse_frac -- доля N под "пачку" цели (короткое окно вокруг t0=2R/c).
    dt         -- такт, с (совпадает с `TactSequence.dt`, если не передан `state`
                  напрямую из уже спроецированного `KinematicsSample`).
    """

    n_samples: int = 1024
    snr_db: float = 12.0
    pulse_frac: float = 0.05
    dt: float = 1.0

    def build(self, state: TargetState, cfg: ProjectConfig, rng: np.random.Generator,
              add_noise: bool = True) -> np.ndarray:
        """`state` -> `(nx, ny, N)` complex64: шум + splat точечной цели по (kx,ky,R).

        `add_noise` (P4/M1): `False` -> без AWGN (амплитуда всё ещё калибрована по
        `snr_db`) -- нужно, когда несколько целей суммируются когерентно и шум должен
        быть добавлен ОДИН раз поверх суммы, см. `add_shared_noise`/`iter_multi_cubes`.
        """
        kinematics = Kinematics(cfg)
        sample = kinematics.project(state, self.dt)
        return self.build_from_sample(sample, cfg, rng, add_noise=add_noise)

    def build_from_sample(self, sample: KinematicsSample, cfg: ProjectConfig,
                           rng: np.random.Generator, add_noise: bool = True) -> np.ndarray:
        """Как `build`, но проекция уже посчитана `Kinematics` (реюз -- не считать дважды за такт).

        `add_noise` -- см. `build` (P4/M1, дефолт `True` = поведение как раньше).
        """
        modulation = _MODULATION_BY_NAME.get(cfg.modulation)
        if modulation is None:
            raise ValueError(f"неизвестная модуляция cfg.modulation={cfg.modulation!r}")
        waveform = WaveformFactory().create(modulation)

        window = self._delay_window(sample.r, cfg.wave.fs)
        spec = WaveformSpec(
            fs=cfg.wave.fs,
            carrier_hz=cfg.wave.carrier_hz,
            n_samples=self.n_samples,
            amplitude=1.0,
            phase=sample.doppler_phase,               # vr -> фазовая прогрессия (задел Доплер)
            fdev_hz=cfg.wave.fdev_hz,
            snr_db=self.snr_db,
            tau_s=2.0 * sample.r / C_LIGHT,             # для GPU-бэкенда (A9-gap1); numpy игнорит
            window=window,
            meta={
                "kx": sample.kx, "ky": sample.ky,
                "nx": float(cfg.array.nx), "ny": float(cfg.array.ny),
            },
            add_noise=add_noise,
        )
        field = waveform.render(NumpyBackend(), spec, rng)
        return field.data

    def _delay_window(self, r_m: float, fs: float) -> TimeWindow:
        """Короткое окно вокруг задержки `t0=2R/c` -- позиция цели по дальности (A9-gap1)."""
        max_t0 = max(0.0, (self.n_samples - 1) / fs)
        t0 = float(np.clip(2.0 * r_m / C_LIGHT, 0.0, max_t0))
        dur = max(1.0 / fs, self.pulse_frac * self.n_samples / fs)
        dur = min(dur, self.n_samples / fs - t0)
        dur = max(dur, 1.0 / fs)
        return TimeWindow(kind="short", t0=t0, dur=dur)

    def add_shared_noise(self, volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Добавляет комплексный AWGN ОДИН раз поверх уже просуммированного объёма (P4/M1).

        Реюз того же бэкенда и той же калибровочной константы `NOISE_POWER`, что
        `render_pipeline` использует внутри `build_from_sample` (per-target,
        `add_noise=True` по умолчанию) -- идентичный шумовой пол, просто наложенный
        один раз на сумму целей, а не N раз (N-кратный избыток мощности шума, М1
        сверки Кодо). Не мутирует `volume`.
        """
        return NumpyBackend().add_noise(volume, NOISE_POWER, rng)


def iter_cubes(tacts: Iterable[Tact], builder: VolumeBuilder, cfg: ProjectConfig,
               rng: np.random.Generator,
               data_context: DataContext | None = None) -> Iterator[tuple[Tact, np.ndarray]]:
    """Связка `TactSequence` (P1, движение) + `VolumeBuilder` (P2, заполнение).

    На каждый такт строит объём из уже посчитанной `tact.sample` (`Kinematics`
    отработала внутри `TactSequence.__next__`, второй раз не считаем) и, если дана
    `data_context`, публикует его в шину под ключом `CUBE_CHANNEL` ("cube") -- визуал
    подписывается Observer'ом (SPEC §4).

    Сознательно НЕ встроено в `TactSequence` (P1): `TactSequence` -- SRP-чистый
    итератор движения, уже покрыт тестами P1 (`tests/test_body_motion.py`), трогать
    его лишний раз ради P2-заполнения -- риск регресса без выгоды. Склейка вынесена
    сюда отдельной функцией уровня Composition Root (реюзуется и демо, и тестами
    P2 -- цикл не дублируется).
    """
    for tact in tacts:
        vol = builder.build_from_sample(tact.sample, cfg, rng)
        if data_context is not None:
            data_context.publish(CUBE_CHANNEL, vol)
        yield tact, vol


def iter_multi_cubes(multi_tacts: Iterable[MultiTact], builder: VolumeBuilder, cfg: ProjectConfig,
                      rng: np.random.Generator,
                      data_context: DataContext | None = None) -> Iterator[tuple[MultiTact, np.ndarray]]:
    """Мульти-целевой аналог `iter_cubes` (P4): `MultiTactSequence` (движение N целей,
    `tact_sequence.py`) + `VolumeBuilder` (заполнение) -> когерентная сумма объёма такта.

    🔴 M1 (сверка Кодо): каждая цель рендерится с `add_noise=False` (см. `VolumeBuilder.
    build_from_sample`) -- N целей суммируются БЕЗ шума, затем `builder.add_shared_noise`
    добавляет AWGN ОДИН раз поверх суммы (не N раз, как было бы при наивном суммировании
    N независимо зашумлённых объёмов -- это завысило бы мощность шума в N раз).

    🔴 M3 (сверка Кодо): когерентная сумма целей -- поэлементное сложение `(nx,ny,N)`-
    массивов НАПРЯМУЮ, без обёртки каждой цели в `SignalSource`/`Scene` (`scene.py`).
    `Scene`-Composite рассчитан на источники своего домена (`grid, RangeConfig, rs ->
    array`, стерео над `ArrayGrid.steering`/`RangeConfig.n_real`) -- ровно так уже
    работают помехи (`SceneModeler`/`jammers.py`). Per-target рендер цели (`VolumeBuilder`,
    домен `WaveformSpec`+`render_pipeline`: своя несущая/fs/окно/steering) уже сам строит
    готовый массив `(nx,ny,N)` -- заворачивать его в адаптер-`SignalSource` ради
    единообразия с `Scene` добавило бы слой косвенности без выгоды (тот же результат, та
    же форма); поэтому цели складываются как numpy-массивы напрямую, а не через Composite.

    Помехи (P3, `SceneModeler.contribute_to`) сюда сознательно НЕ встроены -- по тому же
    принципу разделения ответственности, что и `iter_cubes` (P2) не знает про P3: вызывающий
    код (демо/тесты) применяет `SceneModeler.contribute_to(vol, cfg, jam_rng)` поверх
    результата этой функции, как уже делает `demo_body_motion_jammers.py`.
    """
    for multi_tact in multi_tacts:
        vol: np.ndarray | None = None
        for tact in multi_tact.tacts:
            contrib = builder.build_from_sample(tact.sample, cfg, rng, add_noise=False)
            vol = contrib if vol is None else vol + contrib
        if vol is None:
            raise ValueError("MultiTact.tacts пуст -- нет целей для заполнения объёма")
        vol = builder.add_shared_noise(vol, rng)
        if data_context is not None:
            data_context.publish(CUBE_CHANNEL, vol)
        yield multi_tact, vol
