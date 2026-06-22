# 04 — Testing Python (NO pytest)

> **paths:** `tests/**`, `core/**/tests/**`
> Правило критическое — нарушение = потеря работы Alex (прецедент DSP-GPU).

## 🚫 ЗАПРЕЩЕНО

- `import pytest`
- `pytest.fixture`, `pytest.mark.*`, `pytest.raises`
- `conftest.py`
- `pyproject.toml [tool.pytest.*]`

**Любой `pytest`-вызов в коде = немедленно убрать.**

## ✅ Замена — `TestRunner + SkipTest`

Раннер живёт в `common/runner.py` (перенесён вместе с конфигом).

```python
# tests/test_fft3d.py
from common.runner import TestRunner, SkipTest, AssertionGroup
from core.config import default_scenario
from core.models import Fft3DModel, AxisWindows, HannWindow


class Fft3DTests(TestRunner):

    def setup(self):
        cfg = default_scenario()
        self.model = Fft3DModel(cfg.array, cfg.range,
                                windows=AxisWindows(HannWindow(), HannWindow(), HannWindow()))
        self.cfg = cfg

    def test_cube_shape(self) -> AssertionGroup:
        g = AssertionGroup("fft3d.shape")
        cube = self.model.process(self.cfg)
        g.add(cube is not None, "cube must not be None")
        g.add(cube.data.ndim == 3, "cube must be 3D")
        return g

    def test_skip_if_no_torch(self) -> AssertionGroup:
        try:
            import torch  # noqa: F401
        except ImportError:
            raise SkipTest("torch не установлен — пропускаем CNN-тест")
        # ...


if __name__ == "__main__":
    Fft3DTests().run_all()
```

## Главный runner

```python
# tests/all_test.py
from tests.test_fft3d import Fft3DTests
from tests.test_classifier import ClassifierTests

if __name__ == "__main__":
    for cls in [Fft3DTests, ClassifierTests]:
        cls().run_all()
```

## Запуск

```bash
python tests/all_test.py        # все наборы
python tests/test_fft3d.py      # один набор
```

## Почему НЕ pytest

- fixtures скрывают setup/teardown — отладка сложнее.
- plugins ломаются между версиями.
- mock'и часто маскируют реальные баги.
- TestRunner — наш, контролируем полностью.

## ✅ Allowed dev tools

- `ruff` (lint), `mypy` (type check), `coverage.py` standalone (без pytest-cov).
