"""Агрегатор тест-наборов radar3d. Запуск:  python tests/all_test.py

ВНИМАНИЕ: pytest ЗАПРЕЩЁН (см. .claude/rules/04-testing-python.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Чтобы работала и форма `python tests/all_test.py`, и `python -m tests.all_test`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_smoke import SmokeTests

SUITES = [SmokeTests]


def main() -> int:
    ok = True
    for cls in SUITES:
        ok = cls().run_all() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
