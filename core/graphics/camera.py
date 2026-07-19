"""Единая сцена/камера (Value Object) — ОДНА проекция мир→экран для всех окон.

Мир: (kx азимут, ky угол места, r дальность). Камера — орбитальная (az, el), стоит в
нулевой дальности. Конвенция: +kx → влево, +ky → вверх, r=0 у зрителя (объекты, приближаясь,
летят на него).

**Ровно один метод `project`.** Правое информационное окно (поле kx·ky) — НЕ отдельный
рисовальщик, а ТА ЖЕ сцена, спроецированная вдоль оси дальности (`Projection.field()`:
az=0, el=0 ⇒ дальность уходит перпендикулярно экрану, остаётся плоскость kx·ky). 3D — та же
`project` с наклонным ракурсом. Один источник ⇒ окна согласованы by construction, зеркальность
невозможна. Прецедент/правило: `.claude/rules/07-math-in-core.md`; тесты: `tests/test_camera.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

AZ0 = 0.42             # эталонный азимут 3D-ракурса (объём + kx влево)
EL0 = -0.32            # эталонный наклон (рад): r=0 ближе-ниже → летят на зрителя
FIT_SCENE = 0.70       # вписать повёрнутый куб в кадр
FIT_FIELD = 0.92       # поле заполняет окно плотнее


@dataclass(frozen=True)
class Projection:
    """Камера сцены (Value Object). `az`, `el` — орбита; `fit` — вписывание в кадр.

    `nx`, `ny` — апертура (kx∈[−nx/2,nx/2], ky∈[−ny/2,ny/2]); `n_range` — ось дальности.
    """

    nx: int
    ny: int
    n_range: int
    az: float = AZ0
    el: float = EL0
    fit: float = FIT_SCENE

    @classmethod
    def field(cls, nx: int, ny: int, n_range: int) -> Projection:
        """Правое инфо-окно kx·ky — вид наблюдателя С НУЛЕВОЙ дальности (r=0), глядящего вдоль +r.

        `az=π` (не 0): смотрим со стороны наблюдателя РЛС (r=0), а НЕ с максимальной дальности —
        иначе азимут kx отражается (зеркальная панель). Разворот на 180° вокруг вертикали ⇒
        +kx→вправо, +ky→вверх (естественный вид с нуля). 3D-куб — свой облётный ракурс (az0), не трогается.
        """
        return cls(nx, ny, n_range, az=math.pi, el=0.0, fit=FIT_FIELD)

    def rotated(self, az: float, el: float) -> Projection:
        """Тот же мир, другой ракурс камеры (для интерактивного вращения 3D)."""
        return Projection(self.nx, self.ny, self.n_range, az=az, el=el, fit=self.fit)

    # ── ЕДИНСТВЕННЫЙ метод проекции: мир (kx, ky, r) → экранный ndc (x, y) + глубина ──
    def project(self, kx: float, ky: float, r: float) -> tuple[float, float, float]:
        """(x, y, depth) в ndc [≈−1..1]. x: +kx→влево, y: +ky→вверх; depth растёт с дальностью."""
        ax = kx / (self.nx / 2.0)
        ay = ky / (self.ny / 2.0)
        zc = (r / self.n_range - 0.5) * 2.0             # дальность в [−1,1], zc=−1 у камеры
        ca, sa = math.cos(self.az), math.sin(self.az)
        ce, se = math.cos(self.el), math.sin(self.el)
        xr = ax * ca - zc * sa                           # орбита вокруг вертикали (ky)
        zr = ax * sa + zc * ca
        yr = ay * ce - zr * se                           # наклон вокруг горизонта
        depth = ay * se + zr * ce                        # r=0 ближе (меньше depth)
        return -xr * self.fit, -yr * self.fit, depth     # −x: kx влево; −y: ky вверх

    def as_js(self) -> dict:
        """Параметры для тонкого JS-рендера demo — единый источник (обе камеры из этого VO)."""
        f = Projection.field(self.nx, self.ny, self.n_range)
        return {"nx": self.nx, "ny": self.ny, "nRange": self.n_range,
                "scene": {"az": self.az, "el": self.el, "fit": self.fit},
                "field": {"az": f.az, "el": f.el, "fit": f.fit},
                "az0": AZ0, "el0": EL0}
