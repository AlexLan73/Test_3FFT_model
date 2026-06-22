"""Демо: три сцены со смещением к краю кадра, с архивацией прогона.

Сцены (1 источник на сцену, чтобы форма читалась чисто):
  1. target  -- чистый ответ: точечная цель в углу, у дальнего края дальности;
  2. barrage -- заградительная: шумовая заливка с краевого направления;
  3. comb    -- гребёнка DRFM: зубцы в дальнем полудиапазоне, под краевым углом.

Каждый запуск -> отдельный каталог out/runs/<дата>/<время>/ (прогоны не
затирают друг друга). На сцену: figures/*.png + data/scene_raw.npy +
manifest.yaml (полное описание сцены для воспроизведения + результат).
Composition Root -- здесь.
"""
from __future__ import annotations
import os
import numpy as np

from core.config import edge_scenarios
from core.models import Fft3DModel, AxisWindows, HannWindow, RuleBasedClassifier
from core.graphics import (CubeScatterVisualizer, AngularMapVisualizer,
                           RangeProfileVisualizer, FigureWriter)
from core.data_context import DataContext, RunWorkspace, config_to_dict
from core.controller import SimulationController

OUT = os.environ.get("RADAR_OUT", "./out")
RANGE_LIMIT = 64          # показываем всю дальностную ось -> виден дальний край

# Где у каждой сцены сидит источник (угол) -> на него наводим гейт/срез.
GATES = {
    "target":  (6, 6),
    "barrage": (-7, 6),
    "comb":    (6, -6),
}
WHAT = {
    "target":  "чистый ответ (цель)",
    "barrage": "заградительная (шум)",
    "comb":    "гребёнка DRFM",
}

# Зоны дальности для sanity-check: локализованный источник -> энергия в одной
# зоне; заградительная (шум) -> примерно поровну по всем зонам.
ZONES = {"бл.0-7": (0, 8), "сер.24-39": (24, 40), "дальн.48-63": (48, 64)}


def _zone_fracs(cube, ix, iy) -> dict[str, float]:
    """Доля энергии профиля дальности ячейки (ix,iy) по зонам + уровень бина 0."""
    prof = cube.magnitude[ix, iy, :]
    total = float((prof ** 2).sum()) + 1e-12
    fr = {lbl: round(float((prof[a:b] ** 2).sum() / total), 4)
          for lbl, (a, b) in ZONES.items()}
    fr["бин0_дБ"] = round(float(20.0 * np.log10(prof[0] / prof.max() + 1e-12)), 1)
    return fr


def _run_scene(ws: RunWorkspace, name: str, cfg, model, clf) -> dict:
    """Прогон одной сцены: фигуры + куб + manifest на диск, возврат записи метрик."""
    data = DataContext(root=ws.data_dir(name))
    controller = SimulationController(model=model, data_context=data)
    outcome = controller.run(cfg, save_as="scene_raw")
    cube = outcome.spectral_cube

    gx, gy = GATES[name]
    writer = FigureWriter(ws.figures_dir(name))
    visualizers = {
        "cube_scatter.png": CubeScatterVisualizer(threshold_db=-22,
                                                  range_limit=RANGE_LIMIT),
        "angular_map.png": AngularMapVisualizer(gate_kx=gx, gate_ky=gy,
                                                gate_half=1.5),
        "range_profiles.png": RangeProfileVisualizer(
            cells=[(gx, gy, f"источник ({gx},{gy})")], range_limit=RANGE_LIMIT),
    }
    for fname, vis in visualizers.items():
        writer.write(vis.render(cube), fname)

    verdict = clf.classify(cube)
    ix, iy = cube.index_of_angle(gx, gy)
    zones = _zone_fracs(cube, ix, iy)
    rec = {
        "name": name, "gx": gx, "gy": gy,
        "verdict": verdict.name, "conf": round(verdict.confidence, 2),
        "peak_bin": int(cube.range_profile_db(ix, iy).argmax()),
        "zones": zones,
    }

    # manifest.yaml -- полное описание сцены + результат (воспроизводимо)
    manifest = {
        "run": ws.run_id,
        "scene": name,
        "description": WHAT[name],
        "gate": {"kx": gx, "ky": gy},
        "config": config_to_dict(cfg),
        "result": {
            "verdict": verdict.name,
            "confidence": round(verdict.confidence, 3),
            "peak_range_bin": rec["peak_bin"],
            "energy_zones": zones,
        },
    }
    ws.write_manifest(name, manifest)

    zstr = "  ".join(f"{lbl}={zones[lbl]:.0%}" for lbl in ZONES)
    print(f"  [{name:7s}] классификатор: {verdict}")
    print(f"            пик дальн. бин ~ {rec['peak_bin']}  |  {zstr}  "
          f"|  бин0={zones['бин0_дБ']:+.0f}дБ")
    return rec


def _summary_md(ws: RunWorkspace, records: list[dict]) -> str:
    """Markdown-сводка прогона."""
    head = ("| сцена | что это | угол | вердикт | пик дальн. |"
            + "".join(f" {lbl} |" for lbl in ZONES) + " бин0 |")
    sep = "|" + "---|" * (5 + len(ZONES) + 1)
    rows = []
    for r in records:
        z = r["zones"]
        zcells = "".join(f" {z[lbl]:.0%} |" for lbl in ZONES)
        rows.append(
            f"| **{r['name']}** | {WHAT[r['name']]} | ({r['gx']:+d},{r['gy']:+d}) "
            f"| {r['verdict']} (p={r['conf']:.2f}) | бин {r['peak_bin']} "
            f"|{zcells} {z['бин0_дБ']:+.0f}дБ |")
    return "\n".join([
        f"# 📈 Прогон {ws.run_id}",
        "",
        "> Автоген: `python edge_scenes_demo.py`. Доли — энергия профиля дальности",
        "> в ячейке источника по зонам. Локализованный источник → энергия в одной",
        "> зоне и `бин0` глубоко внизу; заградительная (шум) → ~поровну по зонам,",
        "> `бин0 ≈ 0 дБ`. Параметры каждой сцены → `<сцена>/manifest.yaml`.",
        "> Подробный разбор картинок → `Doc/edge_scenes.md`.",
        "",
        head, sep, *rows, "",
    ])


def main() -> None:
    ws = RunWorkspace(out_root=OUT)
    scenarios = edge_scenarios()
    # геометрия (array/range) одинакова у всех краевых сцен -> одна модель на все
    any_cfg = next(iter(scenarios.values()))
    model = Fft3DModel(any_cfg.array, any_cfg.range,
                       windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()))
    clf = RuleBasedClassifier()

    print(f"Прогон {ws.run_id} -> {ws.base}")
    records = [_run_scene(ws, name, cfg, model, clf)
               for name, cfg in scenarios.items()]
    path = ws.write_summary(_summary_md(ws, records))
    print(f"\nСводка: {path}")


if __name__ == "__main__":
    main()
