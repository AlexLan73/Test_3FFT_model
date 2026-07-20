"""Агрегатор тест-наборов radar3d. Запуск:  python tests/all_test.py

ВНИМАНИЕ: pytest ЗАПРЕЩЁН (см. .claude/rules/04-testing-python.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работала и форма `python tests/all_test.py`, и `python -m tests.all_test`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_aperture_ixj import ApertureIxjTests
from tests.test_arbiter import CodeArbiterTests, CombinedArbiterTests, EdgeArbiterTests
from tests.test_body_motion import (
    KinematicsTests,
    MessageBusTests,
    MotionModelTests,
    ProjectConfigTests,
)
from tests.test_body_motion_jammers import JammerSceneTests
from tests.test_body_motion_multi import MultiTargetTests
from tests.test_anti_barrage_pipeline import AntiBarragePipelineTests, DiagonalLoadingTests
from tests.test_body_motion_volume import VolumeBuilderTests
from tests.test_calibration import CalibrationTests
from tests.test_camera import CameraTests
from tests.test_cfar import CfarTests
from tests.test_clustering import ClusteringTests
from tests.test_generators import GeneratorsTests
from tests.test_graphics import GraphicsTests
from tests.test_integration import FullPipelineIntegrationTests
from tests.test_mvdr import MvdrNullerTests
from tests.test_nuller import NullerTests
from tests.test_peak_refine import PeakRefineTests
from tests.test_raw_queue import FileSourceTests, RawQueueTests
from tests.test_roi_gate import RoiGateTests
from tests.test_runtime import (
    CommandTests,
    PanelAppTests,
    PanelModelTests,
    PanelPublisherTests,
    SceneServerStepTests,
    TickLogTests,
    TransportTests,
    WsReplayTests,
)
from tests.test_smoke import SmokeTests
from tests.test_snr import SnrTests
from tests.test_tokenizer import (
    FeatureSeparationTests,
    OsCfarPfaTests,
    RangeAssemblyTests,
    VolumeTokenizerTests,
)
from tests.test_targeting import BeamTargetingTests, CognitiveCycleTests
from tests.test_tracking import NearestNeighborTrackerTests
from tests.test_waveform_to_cube import AmToCubeTests, LfmToCubeTests, SquareViewTests

SUITES = [SmokeTests, CameraTests, GraphicsTests, NullerTests, CfarTests, ClusteringTests, SnrTests, GeneratorsTests,
          ProjectConfigTests, MessageBusTests, MotionModelTests, KinematicsTests,
          VolumeBuilderTests, LfmToCubeTests, AmToCubeTests, SquareViewTests,
          ApertureIxjTests,
          JammerSceneTests, MultiTargetTests,
          TransportTests, TickLogTests, PanelPublisherTests, WsReplayTests,
          CommandTests, SceneServerStepTests, PanelModelTests, PanelAppTests,
          RawQueueTests, FileSourceTests,
          FeatureSeparationTests, RangeAssemblyTests, VolumeTokenizerTests, OsCfarPfaTests,
          PeakRefineTests,
          CalibrationTests,
          EdgeArbiterTests, CodeArbiterTests, CombinedArbiterTests,
          NearestNeighborTrackerTests,
          BeamTargetingTests, CognitiveCycleTests, RoiGateTests,
          DiagonalLoadingTests, AntiBarragePipelineTests, MvdrNullerTests,
          FullPipelineIntegrationTests]


def main() -> int:
    ok = True
    for cls in SUITES:
        ok = cls().run_all() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
