"""Индекс всех прогонов: собирает out/runs/<дата>/<время>/*/manifest.yaml в таблицу.

Запуск:  python list_runs.py
Вывод:   таблица в консоль + out/runs/index.md (обзор всей серии тестов).
"""
from __future__ import annotations
import os
from pathlib import Path

from core.data_context import from_yaml

OUT = os.environ.get("RADAR_OUT", "./out")
SCENE_ORDER = {"target": 0, "barrage": 1, "comb": 2}    # порядок сцен в группе


def _emitters_brief(emitters: list[dict]) -> str:
    """Короткая сводка источников: тип@угол + ключевой параметр дальности."""
    parts = []
    for e in emitters:
        tag = str(e.get("type", "?")).replace("Spec", "")
        loc = f"@({e.get('kx', 0):+},{e.get('ky', 0):+})"
        for key in ("range_bin", "lead_bin", "power", "chirp_rate"):
            if key in e and e[key] is not None:
                loc += f",{key}={e[key]}"
                break
        parts.append(tag + loc)
    return "; ".join(parts) if parts else "—"


def _collect(out_root: str) -> list[dict]:
    """Прочитать все manifest.yaml (кроме симлинка latest) -> список записей."""
    base = Path(out_root) / "runs"
    records = []
    for mf in sorted(base.glob("*/*/*/manifest.yaml")):
        if "latest" in mf.parts:                            # пропускаем симлинк
            continue
        try:
            d = from_yaml(mf.read_text(encoding="utf-8"))
        except OSError:
            continue
        res = d.get("result", {})
        records.append({
            "run": d.get("run", "?"),
            "scene": d.get("scene", mf.parent.name),
            "emitters": _emitters_brief(
                d.get("config", {}).get("scene", {}).get("emitters", [])),
            "verdict": res.get("verdict", "?"),
            "conf": res.get("confidence", 0.0),
            "peak": res.get("peak_range_bin", "?"),
        })
    records.sort(key=lambda r: (r["run"], SCENE_ORDER.get(r["scene"], 9)))
    return records


def _index_md(records: list[dict]) -> str:
    rows = ["| прогон | сцена | источники | вердикт | пик дальн. |",
            "|---|---|---|---|---|"]
    for r in records:
        rows.append(f"| {r['run']} | **{r['scene']}** | `{r['emitters']}` "
                    f"| {r['verdict']} (p={r['conf']:.2f}) | бин {r['peak']} |")
    return "\n".join([
        "# 🗂️ Индекс прогонов",
        "",
        f"> Автоген: `python list_runs.py`. Всего записей: **{len(records)}**.",
        "> Параметры каждой сцены — в её `manifest.yaml`. Последний прогон → `latest/`.",
        "",
        *rows, "",
    ])


def main() -> None:
    records = _collect(OUT)
    if not records:
        print("Прогонов не найдено. Сначала: python edge_scenes_demo.py")
        return
    print(f"Найдено прогонов-сцен: {len(records)}\n")
    print(f"{'прогон':<19} {'сцена':<8} {'вердикт':<14} {'пик':>5}  источники")
    for r in records:
        print(f"{r['run']:<19} {r['scene']:<8} "
              f"{r['verdict']+f' (p={r['conf']:.2f})':<14} "
              f"{str(r['peak']):>5}  {r['emitters']}")
    path = Path(OUT) / "runs" / "index.md"
    path.write_text(_index_md(records), encoding="utf-8")
    print(f"\nИндекс сохранён: {path}")


if __name__ == "__main__":
    main()
