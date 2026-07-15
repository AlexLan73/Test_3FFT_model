"""Спецификации источников сцены и сама сцена (Value Objects).

Спецификации -- это чистые данные (что разместить), отделённые от их синтеза.
Конкретные классы-источники строит EmitterFactory (см. generators/factory.py).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmitterSpec:
    """Базовая спецификация излучателя/отражателя."""
    kx: float = 0.0          # угловой бин (азимут), 0 = нормаль
    ky: float = 0.0          # угловой бин (угол места)
    amplitude: float = 1.0


@dataclass(frozen=True)
class TargetSpec(EmitterSpec):
    """Истинная точечная цель на дальности range_bin (несёт текущий код)."""
    range_bin: float = 8.0
    phase: float = 0.0


@dataclass(frozen=True)
class DrfmCombSpec(EmitterSpec):
    """Гребёнка ложных целей DRFM: передний фронт + копии ТОЛЬКО позади."""
    lead_bin: float = 8.0
    spacing: float = 6.0
    count: int = 5
    decay: float = 0.85       # спад амплитуды от зубца к зубцу


@dataclass(frozen=True)
class BarrageSpec(EmitterSpec):
    """Заградительная (шумовая) помеха с одного направления: заливка дальности."""
    power: float = 6.0


@dataclass(frozen=True)
class HamEmitterSpec(EmitterSpec):
    """Стороннее излучение (радиолюбитель): после дерампа -- размаз по дальности."""
    chirp_rate: float | None = None   # None -> авто (полный размах по полосе)


@dataclass(frozen=True)
class ThermalNoiseSpec:
    """Тепловой шум приёмника: независим по элементам, без направления."""
    power: float = 0.02


@dataclass(frozen=True)
class JammerFlags:
    """Флаги включения помех такта (P3, `TASK_body_motion_p3.md` п.2/К-сверка).

    Дефолт -- все `False` (обратная совместимость: старые `SceneConfig()` без
    аргументов не меняют поведение). В P6 те же флаги станут командами сокет-панели
    (SPEC §7·S6) -- имена полей совпадают заранее, чтобы не переименовывать позже.
    `cw`/`vfd`/`arc`/`clutter` включают промышленные помехи (`waveforms/jammers_rf.py`,
    сигнал-уровень) -- на момент P3 адаптер под сырой raw-домен `(nx,ny,n_real)`
    (`generators/jammers.py`, A2) не реализован (см. `SceneModeler`, отчёт P3):
    включение этих флагов -- намеренная `NotImplementedError`, а не тихий no-op.
    """
    barrage: bool = False
    comb: bool = False
    ham: bool = False
    cw: bool = False
    vfd: bool = False
    arc: bool = False
    clutter: bool = False


@dataclass(frozen=True)
class SceneConfig:
    """Набор спецификаций, описывающих обстановку (цели + помехи + шум).

    `jammers` (P3) -- флаги включения помех ТАКТА (см. `JammerFlags`), которые
    строит `generators.SceneModeler` (jammers-only `Scene`, без `ThermalNoise` --
    объём такта, P2, уже содержит калиброванный шум, К1 сверки). `barrage_spec`/
    `comb_spec`/`ham_spec` -- отдельные ОПЦИОНАЛЬНЫЕ поля-спеки для этих помех
    (`None` -> `SceneModeler` берёт дефолт самой спеки, например `BarrageSpec()`).

    Реюз `emitters` для этой цели НЕ выбран специально: `emitters` -- уже
    самостоятельный канал (`SceneBuilder`/`Synthesizer`, edge-сценарии
    `simulation_config.py`) со СВОЕЙ семантикой (список -> `Scene` + обязательный
    `ThermalNoise` в конце, `SceneBuilder.build()`). Смешать туда помехи такта
    означало бы либо тащить весь список через `SceneBuilder` (двойной шум, К1),
    либо распознавать "это jammer такта, а не emitter сцены" через `isinstance`
    -- лишняя связность. Три явных optional-поля -- минимальное решение под
    задачу П3 (одна цель + опционально по одной помехе каждого вида на такт).
    """
    emitters: Sequence[EmitterSpec] = field(default_factory=tuple)
    thermal: ThermalNoiseSpec = field(default_factory=ThermalNoiseSpec)
    jammers: JammerFlags = field(default_factory=JammerFlags)
    barrage_spec: BarrageSpec | None = None
    comb_spec: DrfmCombSpec | None = None
    ham_spec: HamEmitterSpec | None = None
