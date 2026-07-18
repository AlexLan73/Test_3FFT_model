"""ROI-гейт детекций (гл.8 целеуказание -> гейт CFAR).

После целеуказания (`BeamCommand` -- куда светить, `beam.py`) детекции CFAR
имеет смысл ограничить ЗОНОЙ ИНТЕРЕСА (ROI) вокруг ожидаемой позиции цели --
убрать остаточные ложные детекции вне ROI (частая проблема при заграде: нуллер
подавляет заград не идеально, CFAR у него под боком иногда всё ещё «стреляет»
в чужих угловых ячейках/дальностях, где никакого целеуказания не было).

`RoiGate` -- Pure Fabrication: чистый фильтр `list[Detection] -> list[Detection]`
по прямоугольному окну (дальность x угол) вокруг `BeamCommand.center_kx/center_ky`
и `BeamCommand.target_r`. Детекция проходит, если попадает в ROI **хотя бы
одного** переданного `BeamCommand` (несколько целей -- несколько ROI, union).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..anti_barrage.cfar import Detection
from .beam import BeamCommand


@dataclass(frozen=True)
class RoiGate:
    """Фильтр детекций по зоне интереса (ROI) вокруг целеуказания.

    Убирает ложные детекции вне окна ожидаемой цели (§8: целеуказание гейтит
    детекцию CFAR) -- в первую очередь остаточные ложные тревоги заграда,
    не относящиеся ни к одной цели, по которой велось целеуказание.

    Parameters
    ----------
    angle_half : полуширина ROI по углу (в единицах `kx`/`ky` куба, те же, что
                 `Detection.kx`/`Detection.ky` и `BeamCommand.center_kx/ky`) --
                 окно прямоугольное (Чебышёв): `max(|dkx|, |dky|) <= angle_half`.
    range_half : полуширина ROI по дальности, в бинах: `|range_bin - target_r| <= range_half`.

    Геометрия ROI -- центр `BeamCommand.center_kx/center_ky` (грубая угловая
    оценка целеуказания, §8.2), а не отдельные `beam_angles` пучка: пучок
    покрывает конус НЕОПРЕДЕЛЁННОСТИ вокруг центра для ОПРОСА (куда светить),
    а ROI-гейт здесь -- окно вокруг того же центра для ФИЛЬТРАЦИИ уже готовых
    CFAR-детекций; оба заведомо используют один и тот же центр конуса.

    Detection несёт собственные угловые поля `kx`/`ky` (значения оси куба,
    те же единицы, что `center_kx/ky`) -- гейтируем по ним напрямую, индексы
    `kx_idx`/`ky_idx` не нужны.
    """

    angle_half: float = 1.0
    range_half: int = 2

    def filter(
        self,
        detections: list[Detection],
        beams: list[BeamCommand],
    ) -> list[Detection]:
        """Оставить детекции, попадающие в ROI хотя бы одного `BeamCommand`.

        Пустой `beams` (нет целеуказания вообще) -> пустой результат: без
        целеуказания нет ROI, который мог бы что-либо пропустить -- гейт
        консервативно отбрасывает все детекции, а не пропускает их не глядя.

        Не мутирует входы (`detections`, `beams`) -- возвращает новый список,
        детерминированный (сохраняет исходный относительный порядок `detections`).
        """
        if not beams:
            return []
        return [d for d in detections if self._in_any_roi(d, beams)]

    def _in_any_roi(self, det: Detection, beams: list[BeamCommand]) -> bool:
        return any(self._in_roi(det, beam) for beam in beams)

    def _in_roi(self, det: Detection, beam: BeamCommand) -> bool:
        if abs(det.range_bin - beam.target_r) > self.range_half:
            return False
        dkx = abs(det.kx - beam.center_kx)
        dky = abs(det.ky - beam.center_ky)
        return max(dkx, dky) <= self.angle_half
