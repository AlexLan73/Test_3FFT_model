"""WaveformFactory — реестр модуляция → волна (Abstract Factory, §5 спеки/таска P1).

⚠️ Почему НЕ `core.generators.factory.EmitterFactory` (несмотря на «не плодить
фабрики», §4.2 спеки): контракты несовместимы.

- `EmitterFactory.create(spec: EmitterSpec) -> SignalSource` — диспетчит по **типу**
  `EmitterSpec`-подкласса и строит куб-уровневый источник, чей `.contribute(grid,
  rng, rs) -> ndarray[nx,ny,n_real]` уже подразумевает **пост-дечирп** представление
  (см. `core/generators/sources.py`).
- `Waveform.render(backend, spec, rng) -> SignalField` — время-доменный генератор
  (сырое время `[nx,ny,n_samples]`), один `WaveformSpec` используется **всеми**
  модуляциями (различие — в `meta`, не в типе класса спеки) — диспетчить по типу
  спеки, как `EmitterFactory`, здесь нечем: тип спеки один и тот же.

Поэтому — тонкий сиблинг-реестр **того же паттерна** (Abstract Factory + registry,
`register`/`create`, тот же стиль ошибок), но с ключом `Modulation` (VO-enum из
`SignalField`) вместо `type[EmitterSpec]`. Не новая архитектура — тот же приём,
адаптированный под другой контракт (обосновано на ревью, см. TASK_signal_generators_p1).
"""
from __future__ import annotations

from collections.abc import Callable

from .am import AmWaveform
from .base import Waveform
from .cw import CwWaveform
from .field import Modulation
from .lfm import LfmWaveform

Builder = Callable[[], Waveform]


class WaveformFactory:
    """Создаёт `Waveform` по типу модуляции (GRASP Creator, тот же приём, что `EmitterFactory`)."""

    def __init__(self) -> None:
        self._builders: dict[Modulation, Builder] = {}
        self._register_defaults()

    def register(self, modulation: Modulation, builder: Builder) -> None:
        self._builders[modulation] = builder

    def create(self, modulation: Modulation) -> Waveform:
        try:
            return self._builders[modulation]()
        except KeyError as exc:
            raise ValueError(f"Нет билдера волны для {modulation!r}") from exc

    def _register_defaults(self) -> None:
        self.register(Modulation.CW, CwWaveform)
        self.register(Modulation.LFM, LfmWaveform)
        self.register(Modulation.AM, AmWaveform)
