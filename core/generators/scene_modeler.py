"""SceneModeler -- тонкий строитель jammers-only `Scene` поверх объёма такта (P3).

`TASK_body_motion_p3.md` п.1 (A1, сверка Кодо): `Scene`/`SceneBuilder`/`Synthesizer`
уже есть (`core/generators/scene.py`) -- второй Composite НЕ плодим. `SceneModeler`
не наследует `Scene`, а лишь наполняет уже существующий `Scene` источниками помех
через уже существующий `EmitterFactory` (`core/generators/factory.py`) и делегирует
`Scene.contribute(grid, rng, rs)` -- сам он Composite-логику не хранит (Builder).

🔴 К1 (сверка Кодо): `SceneBuilder.build()` **всегда** добавляет `ThermalNoise`
(`scene.py`), а объём такта (P2, `VolumeBuilder.build_from_sample`) уже содержит
шум, калиброванный по `snr_db` (`render_pipeline`, `NOISE_POWER=1.0`). Реюзить
`SceneBuilder.build` напрямую нельзя -- получится двойной шум. Поэтому здесь
`Scene()` строится вручную по флагам `SceneConfig.jammers` (`JammerFlags`),
`ThermalNoise` НЕ добавляется никогда (ни явно, ни через `SceneBuilder`).

🔴 К2 (сверка Кодо): `contribute_to` берёт фактическое число отсчётов дальности
`N = volume.shape[2]` объёма такта и строит `RangeConfig(n_real=N, n_fft=N)` --
НЕ дефолтный `cfg.range_` (у того `n_real=16`, под P2-объём `N=1024..10000`
несовместим формой -- сложение с рассинхроном форм было бы битым результатом).

🟡 К3 (сверка Кодо): мощность помех -- ИЗ СПЕКИ (`BarrageSpec.power`,
`DrfmCombSpec.amplitude`/`decay` и т.д.), в этом модуле НЕ хардкодится: если
`SceneConfig.barrage_spec`/`comb_spec`/`ham_spec` не заданы (`None`), берётся
дефолт САМОЙ спеки (`BarrageSpec()` и т.п., см. `core/config/scene_config.py`) --
критерий "цель выживает" сверяется именно на этих дефолтах. Демо (наглядность,
"мощность 20-60") задаёт свои `*_spec` через `SceneConfig`, не через код здесь.

cw/vfd/arc/clutter (п. "промышленные (опц.)"): сигнал-уровневые помехи
(`waveforms/jammers_rf.py`) живут в домене `WaveformSpec`+`render_pipeline`
(нужны fs/несущая/окно/калибровка по snr) -- адаптация под сырой raw-домен
`(nx,ny,n_real)` (`generators/jammers.py`, A2) нетривиальна (иная физика зонда,
иной способ калибровки мощности). Оставлены заглушками: включение флага -> явный
`NotImplementedError` (а не тихий no-op), см. отчёт задачи P3.
"""
from __future__ import annotations

import numpy as np

from ..config import BarrageSpec, DrfmCombSpec, HamEmitterSpec, ProjectConfig, RangeConfig
from .factory import EmitterFactory
from .grid import ArrayGrid
from .scene import Scene

# Флаги, для которых сигнал-уровневый адаптер (waveforms/jammers_rf.py) под
# сырой raw-домен пока не реализован (К-сверка, см. докстринг модуля выше).
_UNIMPLEMENTED_JAMMERS: tuple[str, ...] = ("cw", "vfd", "arc", "clutter")


class SceneModeler:
    """Наполняет jammers-only `Scene` по `SceneConfig.jammers` (Builder, НЕ Composite).

    Не хранит собственный список источников -- на каждый вызов `build_jammers`
    строит новую `Scene` (Scene и так дёшева, состояния между тактами не держим).
    """

    def __init__(self, factory: EmitterFactory | None = None) -> None:
        self._factory = factory or EmitterFactory()

    def build_jammers(self, cfg: ProjectConfig) -> Scene:
        """`Scene`, наполненная ТОЛЬКО помехами (без цели, без `ThermalNoise`, К1)."""
        flags = cfg.scene.jammers
        scene = Scene()

        if flags.barrage:
            spec: BarrageSpec = cfg.scene.barrage_spec if cfg.scene.barrage_spec is not None \
                else BarrageSpec()
            scene.add(self._factory.create(spec))
        if flags.comb:
            comb_spec: DrfmCombSpec = cfg.scene.comb_spec if cfg.scene.comb_spec is not None \
                else DrfmCombSpec()
            scene.add(self._factory.create(comb_spec))
        if flags.ham:
            ham_spec: HamEmitterSpec = cfg.scene.ham_spec if cfg.scene.ham_spec is not None \
                else HamEmitterSpec()
            scene.add(self._factory.create(ham_spec))

        enabled_unimplemented = [name for name in _UNIMPLEMENTED_JAMMERS if getattr(flags, name)]
        if enabled_unimplemented:
            raise NotImplementedError(
                f"Помехи {enabled_unimplemented} (waveforms/jammers_rf.py, сигнал-уровень) "
                "пока не адаптированы под сырой raw-домен (nx,ny,n_real) -- см. отчёт "
                "TASK_body_motion_p3.md (SceneModeler.build_jammers)"
            )
        return scene

    def contribute_to(self, volume: np.ndarray, cfg: ProjectConfig,
                       rs: np.random.Generator) -> np.ndarray:
        """`volume + вклад_помех`, НЕ мутируя `volume` (К2: `RangeConfig` под фактический N).

        При всех флагах `False` вклад -- нулевой массив (`Scene._empty`), результат
        численно равен входу (см. `SignalSource._empty`, `sources.py`).
        """
        grid = ArrayGrid.from_config(cfg.array)
        if volume.shape[:2] != (grid.nx, grid.ny):
            raise ValueError(
                f"volume.shape[:2]={volume.shape[:2]} не совпадает с cfg.array=({grid.nx},{grid.ny})"
            )
        n = volume.shape[2]
        rng = RangeConfig(n_real=n, n_fft=n)   # К2: фактический N объёма, не cfg.range_
        scene = self.build_jammers(cfg)
        contrib = scene.contribute(grid, rng, rs)
        return volume + contrib.astype(volume.dtype)
