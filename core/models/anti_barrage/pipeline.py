"""Единый anti-barrage тракт (§phase2): Facade над готовыми фильтрами.

Связывает SubspaceNuller (угловое подавление помехи, работает в домене
элементов решётки ДО FFT) и CaCfarDetector (детект по дальности, работает
на SpectralCube ПОСЛЕ FFT).  Между ними — доменный разрыв: nuller.apply()
принимает/возвращает сырой куб (nx, ny, K) complex, а cfar.detect() ждёт
SpectralCube (после спектрального преобразования).  Поэтому pipeline берёт
третьим коллаборатором RadarModel (Fft3DModel в проекте) — он и есть
связующее звено (Composition Root передаёт готовый экземпляр, DI).
"""
from __future__ import annotations

import numpy as np

from ..base import RadarModel
from .base import Nuller
from .cfar import CaCfarDetector, Detection


class AntiBarragePipeline:
    """Единый anti-barrage тракт (§phase2): подавить заград по углу
    (Nuller, напр. SubspaceNuller) -> преобразовать в спектральный куб
    (RadarModel, например Fft3DModel) -> детектировать цель по дальности
    (CaCfarDetector).

    Facade над тремя готовыми компонентами (DI, Composition Root в
    main.py/demo связывает конкретные реализации). Вход process() не
    мутируется ни на одном из шагов (это гарантируют nuller.apply и
    fft_model.process по контракту).

    Parameters
    ----------
    nuller : Nuller
        Угловое подавление помехи (Strategy/DIP — см. `Nuller`). В проекте
        подключается `SubspaceNuller` (EVD ковариации, опционально с diagonal
        loading — см. `SubspaceNuller.loading`); `RobustMvdrNuller` соответствует
        тому же контракту, но в pipeline пока не подключается.
    fft_model : RadarModel
        Преобразование очищенного сырого куба (nx, ny, K) в SpectralCube
        (напр. Fft3DModel) — согласующее звено между doменами nuller и cfar.
    cfar : CaCfarDetector
        Детекция целей по дальности на SpectralCube.
    """

    def __init__(
        self,
        nuller: Nuller,
        fft_model: RadarModel,
        cfar: CaCfarDetector,
    ) -> None:
        self._nuller = nuller
        self._fft_model = fft_model
        self._cfar = cfar

    def process(self, datacube: np.ndarray) -> list[Detection]:
        """Прогнать сырой куб через полный anti-barrage тракт.

        Parameters
        ----------
        datacube : np.ndarray
            Сырой куб (nx, ny, K) complex. НЕ мутируется.

        Returns
        -------
        list[Detection]
            Детекции CA-CFAR после углового подавления помехи.
        """
        suppressed = self._nuller.apply(datacube)      # (nx, ny, K), помеха подавлена
        cube = self._fft_model.process(suppressed)      # SpectralCube
        return self._cfar.detect(cube)
