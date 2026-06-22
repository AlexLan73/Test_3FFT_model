#!/usr/bin/env bash
# Hook: PreToolUse(Bash) — защита от опасных команд (Debian)
# exit 2 = заблокировать команду

RAW="$(cat)"
CMD="$(echo "${RAW}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")"

if [[ -z "${CMD}" ]]; then
    exit 0
fi

# Опасные паттерны
DANGEROUS=(
    "git reset --hard"
    "git clean -f"
    "git push --force"
    "git push -f "
    "git branch -D"
    "rm -rf /"
    "rm -rf ~"
    "rm -rf \$HOME"
)

for pat in "${DANGEROUS[@]}"; do
    if [[ "${CMD}" == *"${pat}"* ]]; then
        echo ""
        echo "⛔ [HOOK] ЗАБЛОКИРОВАНО: обнаружена опасная операция!"
        echo "   Команда содержит: '${pat}'"
        echo "   Подтверди явно в чате, если уверен."
        exit 2
    fi
done

# pytest — запрещён (см. rules/04-testing-python.md)
if [[ "${CMD}" =~ (^|[^a-zA-Z])pytest($|[^a-zA-Z]) ]]; then
    echo ""
    echo "🚫 [HOOK] pytest ЗАПРЕЩЁН в проекте — используй python tests/all_test.py (TestRunner)."
    echo "   См. .claude/rules/04-testing-python.md"
    exit 2
fi

exit 0
