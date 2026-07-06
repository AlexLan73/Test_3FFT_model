"""Демо интерактивного 3D-куба (plotly, самодостаточный HTML).

Ветка независима от `main.py`: собирает InteractiveCubeVisualizer + HtmlWriter,
прогоняет эталонный сценарий -> out/figures/cube_interactive.html.
Импорт plotly -- только здесь и в core.graphics.interactive (мягкая зависимость).
"""
from __future__ import annotations

import os

from core.config import default_scenario
from core.controller import SimulationController
from core.data_context import DataContext
from core.graphics.interactive import HtmlWriter, InteractiveCubeVisualizer
from core.models import AxisWindows, Fft3DModel, HannWindow

OUT = os.environ.get("RADAR_OUT", "./out")


def main() -> None:
    cfg = default_scenario()

    model = Fft3DModel(cfg.array, cfg.range,
                       windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()))
    data = DataContext(root=os.path.join(OUT, "data"))
    controller = SimulationController(model=model, data_context=data)

    outcome = controller.run(cfg, save_as="scene_raw_interactive")
    cube = outcome.spectral_cube

    vis = InteractiveCubeVisualizer()
    writer = HtmlWriter(os.path.join(OUT, "figures"))
    path = writer.write(vis.render(cube), "cube_interactive.html")
    print("written", path)


if __name__ == "__main__":
    main()
