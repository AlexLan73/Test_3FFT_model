"""Тесты core/data_context/run_workspace.py -- H2: сериализация jammers/*_spec.

🚫 pytest -- только TestRunner (см. .claude/rules/04-testing-python.md).
Запуск:  python tests/test_run_workspace.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common.runner import AssertionGroup, TestRunner  # noqa: E402
from core.config import (  # noqa: E402
    BarrageSpec,
    DrfmCombSpec,
    HamEmitterSpec,
    JammerFlags,
    ProjectConfig,
    SceneConfig,
    TargetSpec,
)
from core.data_context.run_workspace import (  # noqa: E402
    from_yaml,
    project_config_from_dict,
    project_config_to_dict,
    to_yaml,
)


class RunWorkspaceSceneJammersTests(TestRunner):
    """H2: `jammers`/`barrage_spec`/`comb_spec`/`ham_spec` не теряются на роундтрипе."""

    def test_scene_jammers_and_specs_roundtrip_dict(self) -> AssertionGroup:
        g = AssertionGroup("run_workspace.scene_jammers_specs_roundtrip_dict")
        scene = SceneConfig(
            emitters=(TargetSpec(kx=1.0, ky=2.0, amplitude=0.9, range_bin=10.0, phase=0.3),),
            jammers=JammerFlags(barrage=True, comb=True, ham=False, cw=True),
            barrage_spec=BarrageSpec(kx=3.0, ky=-1.0, amplitude=1.2, power=9.0),
            comb_spec=DrfmCombSpec(kx=0.5, lead_bin=6.0, spacing=4.0, count=3, decay=0.7),
            ham_spec=HamEmitterSpec(kx=-2.0, chirp_rate=1.5e6),
        )
        cfg = ProjectConfig(scene=scene)
        restored = project_config_from_dict(project_config_to_dict(cfg))
        rs = restored.scene

        g.add(rs.jammers.barrage is True and rs.jammers.comb is True
              and rs.jammers.ham is False and rs.jammers.cw is True,
              f"jammers-флаги должны пережить dict-роундтрип, получено {rs.jammers}")
        g.add(rs.barrage_spec is not None and rs.barrage_spec.power == 9.0 and rs.barrage_spec.kx == 3.0,
              f"barrage_spec должен пережить dict-роундтрип, получено {rs.barrage_spec}")
        g.add(rs.comb_spec is not None and rs.comb_spec.count == 3 and rs.comb_spec.decay == 0.7,
              f"comb_spec должен пережить dict-роундтрип, получено {rs.comb_spec}")
        g.add(rs.ham_spec is not None and rs.ham_spec.chirp_rate == 1.5e6,
              f"ham_spec должен пережить dict-роундтрип, получено {rs.ham_spec}")
        g.add(isinstance(rs.barrage_spec, BarrageSpec), "barrage_spec должен восстановиться как BarrageSpec (не base EmitterSpec)")
        g.add(isinstance(rs.comb_spec, DrfmCombSpec), "comb_spec должен восстановиться как DrfmCombSpec")
        g.add(isinstance(rs.ham_spec, HamEmitterSpec), "ham_spec должен восстановиться как HamEmitterSpec")
        return g

    def test_scene_jammers_and_specs_roundtrip_yaml(self) -> AssertionGroup:
        g = AssertionGroup("run_workspace.scene_jammers_specs_roundtrip_yaml")
        scene = SceneConfig(
            jammers=JammerFlags(vfd=True, arc=True, clutter=True),
            barrage_spec=BarrageSpec(power=4.5),
        )
        cfg = ProjectConfig(scene=scene)
        text = to_yaml(project_config_to_dict(cfg))
        restored = project_config_from_dict(from_yaml(text))
        rs = restored.scene

        g.add(rs.jammers.vfd and rs.jammers.arc and rs.jammers.clutter,
              f"jammers-флаги должны пережить YAML-роундтрип, получено {rs.jammers}")
        g.add(rs.barrage_spec is not None and abs(rs.barrage_spec.power - 4.5) < 1e-9,
              f"barrage_spec.power должен пережить YAML-роундтрип, получено {rs.barrage_spec}")
        g.add(rs.comb_spec is None, "comb_spec должен остаться None, если не задан (не появляться из ничего)")
        g.add(rs.ham_spec is None, "ham_spec должен остаться None, если не задан")
        return g

    def test_scene_without_jammers_defaults_roundtrip(self) -> AssertionGroup:
        """Обратная совместимость: старый SceneConfig() без jammers/*_spec не ломается."""
        g = AssertionGroup("run_workspace.scene_defaults_roundtrip")
        cfg = ProjectConfig()
        restored = project_config_from_dict(project_config_to_dict(cfg))
        rs = restored.scene
        g.add(rs.jammers == JammerFlags(), f"дефолтные jammers должны остаться все False, получено {rs.jammers}")
        g.add(rs.barrage_spec is None, "barrage_spec по умолчанию должен остаться None")
        g.add(rs.comb_spec is None, "comb_spec по умолчанию должен остаться None")
        g.add(rs.ham_spec is None, "ham_spec по умолчанию должен остаться None")
        return g


if __name__ == "__main__":
    ok = RunWorkspaceSceneJammersTests().run_all()
    sys.exit(0 if ok else 1)
