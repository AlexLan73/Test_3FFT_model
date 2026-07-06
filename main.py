"""Точка входа: собирает зависимости и прогоняет эталонный сценарий.

Демонстрирует сквозной поток config -> generators -> model -> graphics/data.
Зависимости связываются здесь (Composition Root) и инъектируются в контроллер.
"""
from __future__ import annotations
import os

from core.config import default_scenario
from core.models import Fft3DModel, AxisWindows, HannWindow
from core.graphics import (CubeScatterVisualizer, AngularMapVisualizer,
                           RangeProfileVisualizer, FigureWriter, AxisLayout)
from core.data_context import DataContext
from core.models import RuleBasedClassifier
from core.controller import SimulationController

OUT = os.environ.get("RADAR_OUT", "./out")


def main() -> None:
    cfg = default_scenario()

    # --- composition root: связываем абстракции с конкретикой ---
    model = Fft3DModel(cfg.array, cfg.range,
                       windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()))
    data = DataContext(root=os.path.join(OUT, "data"))
    controller = SimulationController(model=model, data_context=data)

    outcome = controller.run(cfg, save_as="scene_raw")
    cube = outcome.spectral_cube

    # --- визуализация (полиморфно перебираем стратегии) ---
    writer = FigureWriter(os.path.join(OUT, "figures"))
    visualizers = {
        "cube_scatter.png": CubeScatterVisualizer(threshold_db=-22, range_limit=40),
        "cube_scatter_range_depth.png": CubeScatterVisualizer(
            threshold_db=-22, range_limit=40, layout=AxisLayout.range_in_depth()),
        "angular_map.png": AngularMapVisualizer(gate_kx=0, gate_ky=0, gate_half=1.5),
        "range_profiles.png": RangeProfileVisualizer(
            cells=[(0, 0, "ячейка цели (в гейте)"),
                   (5, -3, "ячейка радиолюбителя")], range_limit=40),
    }
    for name, vis in visualizers.items():
        path = writer.write(vis.render(cube), name)
        print("written", path)

    # --- классификация: детерминированный различитель (без torch) ---
    clf = RuleBasedClassifier()
    print("класс сцены:", clf.classify(cube))

    ix, iy = cube.index_of_angle(0, 0)
    lead = cube.range_profile_db(ix, iy)[:40].argmax()
    print(f"передний фронт цели в ячейке (0,0): бин дальности ~ {lead}")


if __name__ == "__main__":
    main()
