"""WaveformToCube -- заполнение куба «угол×угол×дальность» (Strategy, P5, SPEC §2).

Два РАЗНЫХ фронтенда (гл.3 ЛЧМ / гл.4-бис АМ) -- ветвится только заполнение,
дальше выход обеих сводится к общему кубу `SpectralCube` (kx,ky,range) nx×ny×L
(после zero-pad углового FFT — N_pad_x×N_pad_y×L; дефолт 16×16 → 16×16, no-op).

`LfmToCube` (точный, гл.3): ① глобальный дальностный `RangeFft` по всей оси Z
(дечирп + rect FFT, БЕЗ окна) -- ② поячеечный `angular_fft` nx×ny (паддинг до 2ⁿ) с окном Хэмминга
по апертуре -- `|·|` берётся только ПОСЛЕ углового FFT (комплексные данные держим
до этого момента).

`AmToCube` (грубый, гл.4-бис): локальный `Fft3DModel.fftn` по скользящему окну
nx×ny×D, шаг 8/16/32/64 (нахлёст ½ дефолт) -- реюз `Fft3DModel` "как есть" (A6:
для АМ полный 3D-FFT по окну -- ровно то, что он и делает).

⚠️ A9-gap1/патент-фикс инъекции цели для ЛЧМ (не здесь -- см. `build_lfm_target_volume`
в этом же файле): описан подробно в докстринге функции, включая ОТКЛОНЕНИЕ от
буквального знака tau в TASK_body_motion_p5.md (см. там же).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ...config import ProjectConfig
from ...config.array_config import RangeConfig
from ...models.angular_fft import angular_fft
from ...models.fft3d import Fft3DModel
from ...models.range_fft import RangeFft
from ...models.result import Axis, SpectralCube
from ...models.windows import AxisWindows, HammingWindow, HannWindow, WindowFunction
from ...motion import KinematicsSample
from ..backends import NumpyBackend
from ._pipeline import amplitude_for_snr, render_pipeline
from .base import WaveformSpec
from .field import Modulation
from .heterodyne import dechirp
from .placement import TimeWindow
from .reference import getX_numpy

C_LIGHT = 299_792_458.0  # м/с (совпадает с core.motion.kinematics.C_LIGHT)


class WaveformToCube(Protocol):
    """Strategy: сырой объём (nx,ny,N) -> куб `SpectralCube` (kx,ky,range) nx×ny×L (pad 2ⁿ)."""

    def fill(self, volume: np.ndarray, cfg: ProjectConfig) -> SpectralCube: ...


def build_lfm_target_volume(sample: KinematicsSample, cfg: ProjectConfig, n_samples: int,
                             snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Правильная инъекция цели для ЛЧМ-дечирпа (P5-фикс A9-gap1: задержанное ЛЧМ-эхо,
    НЕ короткое временное окно поверх tau=0 чирпа).

    Старый путь (`VolumeBuilder._delay_window`, короткое окно `TimeWindow(kind="short")`
    поверх `getX_numpy(tau=0)`) режет ЧУЖУЮ фазовую траекторию -- вырезанный кусок
    референсного (tau=0) чирпа в позиции t0 имеет фазу φ(t0), а не φ(0) (фаза НАЧАЛА
    чирпа, как у настоящего задержанного эха `s(t-τ)`) -- после дечирпа с полным
    референсом это даёт НЕ константный тон, а фрагмент со своей внутренней частотной
    разверткой -> размазанный пик (51 бин / 637 м, наблюдалось в P2-демо). Правильная
    инъекция -- НАСТОЯЩЕЕ задержанное эхо: тот же чирп, сдвинутый по времени (та же
    формула `reference.getX_numpy`, только с `tau`), тогда дечирп даёт константный тон
    по всему перекрытию -> компактный пик дальностного FFT (~1-2 бина на V1-сетке).

    ⚠️ ОТКЛОНЕНИЕ от буквы TASK_body_motion_p5.md: там указано `tau=+2R/c`. Проверено
    численно (`Read` исходника `reference.getX_numpy`): `in_window = (t>=0)&(t<=ti-dt)`
    с `t = i/fs + tau` -- при `tau>0` окно обрезает ХВОСТ буфера (видимая часть эха
    "уезжает" к НАЧАЛУ записи, а не к позиции t0=2R/c). Чтобы эхо было видно НАЧИНАЯ с
    t0=2R/c и до конца записи (как физически положено -- цель на дальности R видна
    начиная с момента прихода и далее, а не раньше), нужен знак `tau=-2R/c`. Проверено
    end-to-end (дечирп+FFT): совпадает с числами словаря §6 день-в-день (Nfft=4096 ->
    R_est=6002.1 м при R_true=6000 м, ширина -3дБ -> 47..50 м ~ Δr_eff). Одна конвенция
    знака зафиксирована здесь и используется последовательно (`LfmToCube.fill` строит
    `ref=getX_numpy(tau=0)` -- ту же нулевую точку отсчёта).

    Реюз (НЕ дублирование): шаг 1 (формула) -- `reference.getX_numpy`, как `LfmWaveform`;
    шаги 2-5 (окно/раскладка n×n/шум) -- `_pipeline.render_pipeline`, ТОТ ЖЕ код, что
    `LfmWaveform.render` вызывает, просто передаём другой (задержанный) сигнал шага 1 и
    `window=TimeWindow(kind="full")` (доп. маскировка не нужна -- `getX_numpy` уже
    сама секла видимость через `tau`). `VolumeBuilder`/`LfmWaveform` НЕ меняем (риск
    регресса на 91 baseline-тесте, включая `VolumeBuilderTests` P2, которые проверяют
    именно короткое окно как контракт по умолчанию) -- эта функция самостоятельна.
    """
    fs, carrier, fdev = cfg.wave.fs, cfg.wave.carrier_hz, cfg.wave.fdev_hz
    max_t0 = max(0.0, (n_samples - 1) / fs)
    tau = -float(np.clip(2.0 * sample.r / C_LIGHT, 0.0, max_t0))

    spec = WaveformSpec(
        fs=fs, carrier_hz=carrier, n_samples=n_samples, amplitude=1.0,
        phase=sample.doppler_phase, fdev_hz=fdev, snr_db=snr_db, tau_s=tau,
        window=TimeWindow(kind="full"),
        meta={"kx": sample.kx, "ky": sample.ky,
              "nx": float(cfg.array.nx), "ny": float(cfg.array.ny)},
    )
    amplitude = amplitude_for_snr(spec)
    echo_1d = getX_numpy(fs, n_samples, carrier, amplitude, sample.doppler_phase, fdev, 1.0, tau=tau)
    signal_field = render_pipeline(NumpyBackend(), spec, rng, echo_1d, Modulation.LFM)
    return signal_field.data


def build_pulse_echo_volume(
    modulation: Modulation,
    *,
    fs: float,
    carrier_hz: float,
    n_samples: int,
    dur_samples: int,
    t0_samples: int,
    kx: float,
    ky: float,
    nx: int,
    ny: int,
    rng: np.random.Generator,
    amplitude: float = 1.0,
    extra_meta: dict[str, float] | None = None,
) -> np.ndarray:
    """Правильное ЭХО импульсного зонда: задержанная копия `s(t − t0)` (спека ex3 §2, S1).

    Тот же принцип, что `build_lfm_target_volume` выше (A9-gap1): «окно поверх
    глобальной волны режет ЧУЖУЮ фазовую траекторию». Для АМ это огибающая:
    `AmWaveform` считает `1 + m·cos(2π·f_m·t)` от абсолютного t ⇒ окно в позиции t0
    стартует с ПРОИЗВОЛЬНОЙ фазы огибающей. Настоящее эхо — копия импульса,
    сдвинутая по времени: огибающая начинается С ФРОНТА (cos=+1 ⇒ |эхо|=A·(1+m)
    на первом отсчёте). Для CW разница — лишь постоянный фазовый сдвиг несущей.

    Реализация (формулы НЕ дублируем):
      1) импульс рендерится НА НУЛЕ (`TimeWindow(short, t0=0, dur)`) через
         `WaveformFactory` на решётке 1×1 → 1D зонд;
      2) сдвиг на `t0_samples` БЕЗ заворота (хвост за N обрезается нулями);
      3) steering-раскладка nx×ny — реюз `render_pipeline` (шаги 2-5 §4.0) с уже
         сдвинутым 1D сигналом, `window=full`, `add_noise=False` (мульти-цель:
         шум добавляется ОДИН раз поверх суммы, снаружи).
    """
    if not (0 <= t0_samples < n_samples):
        raise ValueError(f"t0_samples={t0_samples} должен быть в [0, {n_samples})")
    dur = max(1, min(int(dur_samples), n_samples))
    meta_zero: dict[str, float] = {"nx": 1.0, "ny": 1.0, "kx": 0.0, "ky": 0.0}
    if extra_meta:
        meta_zero.update(extra_meta)
    spec_zero = WaveformSpec(
        fs=fs, carrier_hz=carrier_hz, n_samples=n_samples, amplitude=amplitude,
        window=TimeWindow(kind="short", t0=0.0, dur=dur / fs),
        meta=meta_zero, add_noise=False,
    )
    from .factory import WaveformFactory  # локальный импорт: разрыв цикла factory↔waveform_to_cube

    pulse_1d = WaveformFactory().create(modulation).render(
        NumpyBackend(), spec_zero, rng,
    ).data[0, 0, :]

    echo_1d = np.roll(pulse_1d, t0_samples)
    if t0_samples > 0:                       # np.roll заворачивает хвост в начало — эхо так не умеет
        echo_1d[:t0_samples] = 0.0
    end = t0_samples + dur
    if end < n_samples:                      # страховка: за концом импульса — нули
        echo_1d[end:] = 0.0

    meta_full: dict[str, float] = {"nx": float(nx), "ny": float(ny),
                                   "kx": float(kx), "ky": float(ky)}
    if extra_meta:
        meta_full.update(extra_meta)
    spec_full = WaveformSpec(
        fs=fs, carrier_hz=carrier_hz, n_samples=n_samples, amplitude=amplitude,
        window=TimeWindow(kind="full"), meta=meta_full, add_noise=False,
    )
    return render_pipeline(NumpyBackend(), spec_full, rng, echo_1d, modulation).data


@dataclass(frozen=True)
class LfmToCube:
    """ЛЧМ-фронтенд (точный, гл.3 §3.2): 2 раздельных FFT, БЕЗ скользящего окна.

    `range_fft`      -- глобальный дальностный FFT (rect, zero-pad `pad_factor`).
    `aperture_window`-- тэйпер апертуры для углового FFT (SPEC §5: Хэмминг обе оси).
    """

    range_fft: RangeFft = field(default_factory=RangeFft)
    aperture_window: WindowFunction = field(default_factory=HammingWindow)

    def fill(self, volume: np.ndarray, cfg: ProjectConfig) -> SpectralCube:
        n = volume.shape[2]
        fs, carrier, fdev = cfg.wave.fs, cfg.wave.carrier_hz, cfg.wave.fdev_hz

        ref = getX_numpy(fs, n, carrier, 1.0, 0.0, fdev, 1.0, tau=0.0)
        dechirped = dechirp(volume, ref)

        mu = fdev / (n / fs)
        range_domain, r_axis = self.range_fft.transform(dechirped, fs, mu)

        angular = angular_fft(range_domain, aperture_window=self.aperture_window)
        magnitude = np.abs(angular)

        # угловой FFT паддит нулями до 2ⁿ по каждой оси независимо (F9) -- оси kx/ky
        # строим на padded-размерах, иначе длина оси не сойдётся с формой спектра
        pow2x, pow2y = cfg.array.padded_shape()
        kx = Axis("kx", np.arange(-pow2x // 2, pow2x // 2), centered=True)
        ky = Axis("ky", np.arange(-pow2y // 2, pow2y // 2), centered=True)
        rng_axis = Axis("range", r_axis, centered=False)
        return SpectralCube(magnitude, kx, ky, rng_axis)


_DEFAULT_AM_WINDOWS = AxisWindows(HammingWindow(), HammingWindow(), HannWindow())


@dataclass(frozen=True)
class AmToCube:
    """АМ-фронтенд (грубый, гл.4-бис): локальный 3D-FFT по скользящему окну nx×ny×D.

    `depth`  -- D, 16..256 (дефолт 16).
    `step`   -- шаг скольжения 8/16/32/64 (дефолт 8 -> нахлёст 50% при depth=16).
    `start`  -- позиция ПЕРВОГО (и единственного для `fill`) под-куба -- Protocol
                `fill()` возвращает ОДИН куб (контракт `WaveformToCube`); полный скан
                внахлёст -- `scan()` (доп. метод, не часть общего протокола).
    `windows`-- тэйперы для реюза `Fft3DModel` (дефолт: Хэмминг апертура + Hann range).
    """

    depth: int = 16
    step: int = 8
    start: int = 0
    windows: AxisWindows = field(default_factory=lambda: _DEFAULT_AM_WINDOWS)

    def __post_init__(self) -> None:
        if not (16 <= self.depth <= 256):
            raise ValueError(f"depth должен быть в [16,256], получено {self.depth}")
        if self.step not in (8, 16, 32, 64):
            raise ValueError(f"step должен быть одним из (8,16,32,64), получено {self.step}")
        if self.start < 0:
            raise ValueError(f"start не может быть отрицательным, получено {self.start}")

    def _model(self, cfg: ProjectConfig) -> Fft3DModel:
        range_cfg = RangeConfig(n_real=self.depth, n_fft=self.depth)
        return Fft3DModel(cfg.array, range_cfg, windows=self.windows)

    def fill(self, volume: np.ndarray, cfg: ProjectConfig) -> SpectralCube:
        """Один под-куб `depth` начиная с `self.start` (Protocol-контракт -- один Cube)."""
        n = volume.shape[2]
        stop = min(n, self.start + self.depth)
        window = volume[:, :, self.start:stop]
        return self._model(cfg).process(window)

    def scan(self, volume: np.ndarray, cfg: ProjectConfig) -> list[tuple[int, SpectralCube]]:
        """Полный скан внахлёст (шаг `self.step`) -- список `(старт_индекс, под-куб)`.

        Возможен произвольный под-куб любого размера/позиции -- хвост (последнее окно)
        может быть короче `depth` ("только конец", SPEC): `Fft3DModel._transform` сам
        зеро-паддит через `s=(nx,ny,n_fft)`, укороченный хвост не роняет обработку.
        """
        n = volume.shape[2]
        model = self._model(cfg)
        results: list[tuple[int, SpectralCube]] = []
        pos = self.start
        while pos < n:
            stop = min(n, pos + self.depth)
            window = volume[:, :, pos:stop]
            results.append((pos, model.process(window)))
            pos += self.step
        return results
