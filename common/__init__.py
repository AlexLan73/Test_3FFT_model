"""common — общая инфраструктура radar3d (test runner — замена pytest, rule 04)."""

from .runner import AssertionGroup, SkipTest, TestRunner

__all__ = ["AssertionGroup", "SkipTest", "TestRunner"]
