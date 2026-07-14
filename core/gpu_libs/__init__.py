"""core.gpu_libs — копии GPU `.so` DSP-GPU (cp313) + тонкий loader (R1, спека §2.2.1).

`.so`-бинарники сюда НЕ коммитятся (см. `.gitignore` в корне репо, `sync_gpu_libs.sh`
их синкает из `DSP-GPU/DSP/Python/libs/`). В git — только `loader.py`, `configGPU.json`,
`sync_gpu_libs.sh`, этот `__init__.py`.
"""
from __future__ import annotations
