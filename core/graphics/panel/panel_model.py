"""panel_model -- GUI-free дата-модель панели: `Field`/`Cell`/`SignalBlock` + закладка +-N (P6, N5).

🟡 N5 (сверка Кодо): логика панели -- ОТДЕЛЬНО от dearpygui (SRP + тестируемость
без дисплея). Этот модуль НЕ импортирует dearpygui и ничего не рисует -- он только
превращает `SpectralCube` (P5, `core.models.result`) + примитивы канала `tracks`
(см. `core/runtime/scene_server.py::_tracks_payload`) в готовые к отрисовке
структуры. `panel_app.py` (тонкая обвязка dearpygui) читает эти структуры и
рисует, сам ничего не считает.

Пирамида типов (по образцу `Field/Cell/Element` из `PyPanelAntennas`, N6 -- образец
недоступен в среде, схема написана по описанию задачи, не скопирована):

  `Cell`   -- одна клетка квадрата 16x16: нормированное значение (0..1) для colormap.
  `Field`  -- ОДНА плоскость дальности (range-бин): сетка `Cell` (nx*ny) + номер бина.
  `SignalBlock` -- закладка +-N ОДНОГО сигнала (цель ИЛИ заград): "3 ряда" (SPEC §5):
      1) `fields`   -- теплокарты плоскостей K-N..K+N вокруг сигнала,
      2) `location` -- (ix,iy), где сигнал сидит в квадрате 16x16,
      3) `tokens`   -- точки-токены рядом с сигналом (переиспользован `SquareToken`,
                       P5, `core/graphics/square_view.py` -- второй токенайзер не плодим).

`lerp_field` -- линейная интерполяция двух `Field` одинаковой формы (для плавной
GUI-анимации между тактами, тот же приём, что "лёгкий lerp" описания задачи).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...models.result import SpectralCube
from ..square_view import SquareToken, SquareView


@dataclass(frozen=True)
class Cell:
    """Одна клетка квадрата 16x16: позиция + нормированное (0..1) значение."""

    ix: int
    iy: int
    value: float


@dataclass(frozen=True)
class Field:
    """Одна плоскость дальности: сетка `Cell` (row-major `ix*ny+iy`) + номер бина."""

    range_bin: int
    nx: int
    ny: int
    cells: tuple[Cell, ...]

    def cell(self, ix: int, iy: int) -> Cell:
        return self.cells[ix * self.ny + iy]

    def as_grid(self) -> np.ndarray:
        """`(nx, ny)` numpy-массив значений -- удобно для colormap-раскраски в GUI."""
        grid = np.zeros((self.nx, self.ny), dtype=np.float64)
        for c in self.cells:
            grid[c.ix, c.iy] = c.value
        return grid


def _field_from_plane(plane: np.ndarray, range_bin: int) -> Field:
    """`(nx,ny)` срез магнитуды -> `Field`, значения нормированы на max плоскости."""
    nx, ny = plane.shape
    vmax = float(plane.max())
    scale = 1.0 / vmax if vmax > 1e-12 else 1.0
    cells = tuple(
        Cell(ix=ix, iy=iy, value=float(plane[ix, iy]) * scale)
        for ix in range(nx) for iy in range(ny)
    )
    return Field(range_bin=range_bin, nx=nx, ny=ny, cells=cells)


def lerp_field(a: Field, b: Field, t: float) -> Field:
    """Линейная интерполяция `a`->`b`, `t` в [0,1] (для плавной GUI-анимации между тактами)."""
    if (a.nx, a.ny) != (b.nx, b.ny):
        raise ValueError(f"lerp_field: несовпадающая форма {(a.nx, a.ny)} vs {(b.nx, b.ny)}")
    t = float(np.clip(t, 0.0, 1.0))
    cells = tuple(
        Cell(ix=ca.ix, iy=ca.iy, value=ca.value * (1.0 - t) + cb.value * t)
        for ca, cb in zip(a.cells, b.cells, strict=True)
    )
    range_bin = int(round(a.range_bin * (1.0 - t) + b.range_bin * t))
    return Field(range_bin=range_bin, nx=a.nx, ny=a.ny, cells=cells)


@dataclass(frozen=True)
class SignalBlock:
    """Закладка +-N одного сигнала -- "3 ряда" (SPEC §5): fields / location / tokens.

    `is_jammer=True` для заградительного блока (SPEC: "заград -- отдельный блок с
    пометкой углов") -- тогда `angle_kx`/`angle_ky` заполнены (позиция заграда на
    угловой карте), а `fields`/`tokens` -- по представительной плоскости (заград
    "размазан" по дальности, единой пиковой плоскости у него физически нет).
    """

    label: str
    fields: tuple[Field, ...]
    location: tuple[int, int]
    tokens: tuple[SquareToken, ...]
    is_jammer: bool = False
    angle_kx: float | None = None
    angle_ky: float | None = None
    slice_tokens: tuple[dict, ...] = ()
    verdict: str | None = None


class PanelModel:
    """GUI-free приёмник кадров сцены + закладка +-N плоскостей (Information Expert).

    `ingest_cube`/`ingest_tracks`/`ingest_tokens` кладут ПОСЛЕДНИЙ полученный кадр
    (панель рисует "живьём", не копит историю -- history/lerp между двумя
    последними тактами делает `panel_app.py` через `lerp_field`, эта модель хранит
    только "текущее"). `ingest_tokens` (S5) -- канал 'tokens' (`SceneServer._tokens_payload`),
    используется `signal_blocks()` для доп. полей `SignalBlock.slice_tokens`/`verdict`.
    """

    def __init__(self, neighbor_planes: int = 5,
                 reduce_mode: Literal["max", "sum"] = "max") -> None:
        self._view = SquareView(reduce_mode=reduce_mode, neighbor_planes=neighbor_planes)
        self._cube: SpectralCube | None = None
        self._tact: int = -1
        self._targets: list[dict] = []
        self._jammers: list[dict] = []
        self._tokens: list[dict] = []
        self._verdicts: list[dict] = []

    # -- закладка +-N -------------------------------------------------------
    @property
    def neighbor_planes(self) -> int:
        return self._view.neighbor_planes

    def set_neighbor_planes(self, n: int) -> None:
        if n < 0:
            raise ValueError(f"neighbor_planes не может быть отрицательным, получено {n}")
        self._view = SquareView(reduce_mode=self._view.reduce_mode, neighbor_planes=n)

    # -- приём кадров ---------------------------------------------------------
    @property
    def tact(self) -> int:
        return self._tact

    @property
    def cube(self) -> SpectralCube | None:
        return self._cube

    def ingest_cube(self, tact: int, cube: SpectralCube) -> None:
        self._tact = tact
        self._cube = cube

    def ingest_tracks(self, tact: int, targets: list[dict], jammers: list[dict] | None = None) -> None:
        self._tact = tact
        self._targets = list(targets)
        self._jammers = list(jammers or [])

    def ingest_tokens(self, tact: int, tokens: list[dict], verdicts: list[dict] | None = None) -> None:
        """Приём канала 'tokens' (`SceneServer._tokens_payload`, S5) -- по образцу `ingest_tracks`."""
        self._tact = tact
        self._tokens = list(tokens)
        self._verdicts = list(verdicts or [])

    # -- производные структуры для GUI --------------------------------------------
    def full_square(self) -> np.ndarray | None:
        """Квадрат 16x16 (reduce по дальности), как публикует `SceneServer` (канал 'squares')."""
        if self._cube is None:
            return None
        return self._view.reduce_square(self._cube)

    def _fields_around(self, cube: SpectralCube, iz: int) -> tuple[Field, ...]:
        block = self._view.neighbor_block(cube, iz)  # (nx,ny,<=2N+1)
        lo = max(0, iz - self.neighbor_planes)
        return tuple(_field_from_plane(block[:, :, k], lo + k) for k in range(block.shape[2]))

    def _slice_tokens_near(self, cube: SpectralCube, ix: int, iy: int) -> tuple[dict, ...]:
        """`self._tokens` (канал 'tokens', S5), у которых хотя бы один пик попал в окно +-1 около (ix,iy)."""
        near: list[dict] = []
        for tok in self._tokens:
            for peak in tok.get("peaks", []):
                pix, piy = cube.index_of_angle(float(peak["kx"]), float(peak["ky"]))
                if abs(pix - ix) <= 1 and abs(piy - iy) <= 1:
                    near.append(tok)
                    break
        return tuple(near)

    def _verdict_at(self, cube: SpectralCube, ix: int, iy: int) -> str | None:
        """`kind` первого `self._verdicts`, чей угол совпадает с (ix,iy) сигнала (S5), иначе `None`."""
        for verdict in self._verdicts:
            vix, viy = cube.index_of_angle(float(verdict["kx"]), float(verdict["ky"]))
            if vix == ix and viy == iy:
                return str(verdict.get("kind"))
        return None

    def signal_blocks(self, threshold_db: float = -10.0) -> list[SignalBlock]:
        """Один `SignalBlock` на цель (`tracks`) + один сводный на активные заграды.

        Блок на сигнал = кол-ву сигналов (критерий приёмки TASK P6): N целей ->
        N target-блоков, `+1` jammer-блок, если в `tracks.jammers` есть хотя бы одна
        запись (иначе список пуст -- заград выключен).
        """
        cube = self._cube
        if cube is None:
            return []
        n_range = cube.magnitude.shape[2]
        all_tokens = self._view.tokenize(cube, threshold_db=threshold_db)
        blocks: list[SignalBlock] = []

        for target in self._targets:
            ix, iy = cube.index_of_angle(float(target["kx"]), float(target["ky"]))
            iz = int(round(float(target.get("range_bin", 0.0))))
            iz = max(0, min(n_range - 1, iz))
            tokens = tuple(tok for tok in all_tokens if abs(tok.ix - ix) <= 1 and abs(tok.iy - iy) <= 1)
            blocks.append(SignalBlock(
                label=f"цель #{target.get('id', len(blocks) + 1)}",
                fields=self._fields_around(cube, iz), location=(ix, iy), tokens=tokens,
                slice_tokens=self._slice_tokens_near(cube, ix, iy),
                verdict=self._verdict_at(cube, ix, iy),
            ))

        if self._jammers:
            j = self._jammers[0]
            ix, iy = cube.index_of_angle(float(j["kx"]), float(j["ky"]))
            # Заград размазан по всей дальности (SPEC: "заливка дальности") -- нет
            # единственного пикового бина, берём глобальный argmax КУБА как
            # представительную плоскость закладки (упрощение, см. финальный отчёт).
            _gx, _gy, iz = self._view.argmax_range(cube)
            tokens = tuple(tok for tok in all_tokens if abs(tok.ix - ix) <= 1 and abs(tok.iy - iy) <= 1)
            kinds = ", ".join(sorted({str(x.get("kind", "?")) for x in self._jammers}))
            blocks.append(SignalBlock(
                label=f"заград ({kinds})", fields=self._fields_around(cube, iz), location=(ix, iy),
                tokens=tokens, is_jammer=True, angle_kx=float(j["kx"]), angle_ky=float(j["ky"]),
                slice_tokens=self._slice_tokens_near(cube, ix, iy),
                verdict=self._verdict_at(cube, ix, iy),
            ))
        return blocks
