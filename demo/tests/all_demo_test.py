"""Агрегатор тестов-приёмки demo/ (🚫 pytest, правило 04).

Расширяемый список: по мере добавления ex2+ дописываем их `*Tests`-наборы.

Запуск:  .venv/Scripts/python.exe demo/tests/all_demo_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работала форма `python demo/tests/all_demo_test.py` (конвенция репо).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from demo.tests.test_denoise import Ex1DenoiseTests  # noqa: E402
from demo.tests.test_ex2 import Ex2AmSquareTests  # noqa: E402
from demo.tests.test_ex3 import Ex3EchoJammersTests  # noqa: E402
from demo.tests.test_ex4 import Ex4FlightTests  # noqa: E402
from demo.tests.test_ex4_server import Ex4ServerTests  # noqa: E402
from demo.tests.test_ex5 import Ex5PeakRefineTests, Ex5WebTests  # noqa: E402
from demo.tests.test_examples import Ex1AmLineTests  # noqa: E402
from demo.tests.test_live_demo import LiveDemoTests  # noqa: E402
from demo.tests.test_matched import Ex1MatchedTests  # noqa: E402
from demo.tests.test_replay_pipeline import ReplayPipelineTests  # noqa: E402
from demo.tests.test_stft import Ex1StftTests  # noqa: E402

_SUITES = [Ex1AmLineTests, Ex1DenoiseTests, Ex1StftTests, Ex1MatchedTests, Ex2AmSquareTests,
           Ex3EchoJammersTests, Ex4FlightTests, Ex4ServerTests, LiveDemoTests, ReplayPipelineTests,
           Ex5PeakRefineTests, Ex5WebTests]


def main() -> bool:
    ok = True
    for cls in _SUITES:
        ok = cls().run_all() and ok
    print(f"\n=== demo/ агрегатор: {'ВСЁ ЗЕЛЁНОЕ' if ok else 'ЕСТЬ ПАДЕНИЯ'} ===")
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
