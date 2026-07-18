"""demo/run_all.py — Composition Root: прогнать все примеры серии, собрать отчёты.

Список `examples` расширяется по мере добавления ex2+ (наследуют `DemoRunner`,
см. `demo/core/runner.py`). Пока в серии только ex1.

Запуск:
    .venv/Scripts/python.exe demo/run_all.py
    .venv/Scripts/python.exe demo/run_all.py --only ex1_am_line --seed 11
    .venv/Scripts/python.exe demo/run_all.py --no-save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Чтобы работала форма `python demo/run_all.py` (как tests/all_test.py, конвенция репо).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from demo.core import DemoRunner  # noqa: E402
from demo.ex1_am_line.denoise import Ex1Denoise
from demo.ex1_am_line.example import Ex1AmLine
from demo.ex1_am_line.matched import Ex1Matched
from demo.ex1_am_line.stft_detect import Ex1StftDetect
from demo.ex2_am_square.example import Ex2AmSquare


def _build_examples() -> list[DemoRunner]:
    return [Ex1AmLine(), Ex1Denoise(), Ex1StftDetect(), Ex1Matched(), Ex2AmSquare()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогнать примеры demo/ (Composition Root).")
    parser.add_argument("--only", type=str, default=None,
                        help="прогнать только пример с этим именем (demo.name)")
    parser.add_argument("--seed", type=int, default=None, help="переопределить seed для всех примеров")
    parser.add_argument("--no-save", action="store_true", help="не писать PNG на диск")
    args = parser.parse_args()

    examples = _build_examples()
    if args.only is not None:
        examples = [ex for ex in examples if ex.name == args.only]
        if not examples:
            raise SystemExit(f"пример {args.only!r} не найден")
    if args.seed is not None:
        for ex in examples:
            ex.seed = args.seed

    for ex in examples:
        report = ex.run(save=not args.no_save)
        print(report)


if __name__ == "__main__":
    main()
