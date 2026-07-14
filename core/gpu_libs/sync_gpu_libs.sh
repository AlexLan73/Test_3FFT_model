#!/usr/bin/env bash
# sync_gpu_libs.sh — копирует собранные DSP-GPU .so (cp313) в core/gpu_libs/ (G12, спека §2.2.1).
#
# .so НЕ коммитятся в git (несколько МБ каждый, замусорили бы историю) — этот
# скрипт синкает их из соседнего репо DSP-GPU при необходимости (пересборка,
# новая машина). В git — только loader.py + configGPU.json + этот скрипт.
#
# Запуск:  core/gpu_libs/sync_gpu_libs.sh [путь_к_DSP-GPU/DSP/Python/libs]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-/home/alex/DSP-GPU/DSP/Python/libs}"

if [[ ! -d "$SRC" ]]; then
    echo "ОШИБКА: не найден каталог с .so: $SRC" >&2
    exit 1
fi

MODULES=(dsp_core dsp_signal_generators dsp_heterodyne dsp_radar dsp_spectrum)
copied=0
for mod in "${MODULES[@]}"; do
    found=$(find "$SRC" -maxdepth 1 -name "${mod}.cpython-313-*.so" | head -n1)
    if [[ -z "$found" ]]; then
        echo "ПРОПУСК: ${mod}.cpython-313-*.so не найден в $SRC" >&2
        continue
    fi
    cp -v "$found" "$SCRIPT_DIR/"
    copied=$((copied + 1))
done

echo "Скопировано $copied/${#MODULES[@]} .so в $SCRIPT_DIR"
echo "configGPU.json — уже в git (core/gpu_libs/configGPU.json), не трогаем."
