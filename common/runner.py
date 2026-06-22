"""Минимальный test runner для rag-mentor — замена pytest (rule 04, ЗАПРЕЩЁН pytest).

Использование:

    from common.runner import TestRunner, SkipTest, AssertionGroup

    class MyTests(TestRunner):
        def setup(self) -> None:        # опционально, вызывается перед каждым test_*
            self.x = 1

        def test_something(self) -> AssertionGroup:
            g = AssertionGroup("my.something")
            g.add(self.x == 1, "x must be 1")
            return g

    if __name__ == "__main__":
        MyTests().run_all()

`run_all()` возвращает True если все группы зелёные (для агрегатора в tests/all_test.py).
"""

from __future__ import annotations

import traceback
from collections.abc import Callable


class SkipTest(Exception):
    """Подними внутри test_*, чтобы пропустить (например, БД не поднята)."""


class AssertionGroup:
    """Набор проверок одного теста. `add(cond, msg)` копит провалы, не падая сразу."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.failures: list[str] = []
        self.passed = 0

    def add(self, condition: bool, message: str) -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(message)

    @property
    def ok(self) -> bool:
        return not self.failures


class TestRunner:
    """Базовый класс. Наследники определяют `test_*` методы, возвращающие AssertionGroup."""

    def setup(self) -> None:  # noqa: B027 — намеренно пустой хук
        """Переопредели для подготовки перед каждым test_*."""

    def _test_methods(self) -> list[tuple[str, Callable[[], AssertionGroup]]]:
        names = sorted(n for n in dir(self) if n.startswith("test_"))
        return [(n, getattr(self, n)) for n in names]

    def run_all(self) -> bool:
        cls = type(self).__name__
        total_ok = total_fail = total_skip = 0
        print(f"\n=== {cls} ===")
        for name, method in self._test_methods():
            try:
                self.setup()
                group = method()
            except SkipTest as exc:
                total_skip += 1
                print(f"  ⏭  {name}: SKIP — {exc}")
                continue
            except Exception:  # noqa: BLE001 — runner ловит всё, печатает traceback
                total_fail += 1
                print(f"  ❌ {name}: ERROR")
                print(traceback.format_exc())
                continue

            if group is None:
                total_ok += 1
                print(f"  ✅ {name}")
            elif group.ok:
                total_ok += 1
                print(f"  ✅ {name} ({group.passed} checks)")
            else:
                total_fail += 1
                print(f"  ❌ {name} ({group.name}): {len(group.failures)} fail / {group.passed} ok")
                for msg in group.failures:
                    print(f"       - {msg}")

        print(f"--- {cls}: {total_ok} ok · {total_fail} fail · {total_skip} skip ---")
        return total_fail == 0
