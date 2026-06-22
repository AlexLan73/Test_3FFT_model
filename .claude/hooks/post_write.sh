#!/usr/bin/env bash
# Hook: PostToolUse(Write) — напоминания при изменении ключевых файлов radar3d (Debian)

RAW="$(cat)"
FILE_PATH="$(echo "${RAW}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null || echo "")"

if [[ -z "${FILE_PATH}" ]]; then
    exit 0
fi

# CLAUDE.md изменён
if [[ "${FILE_PATH}" =~ CLAUDE\.md$ ]]; then
    echo ""
    echo "📝 [HOOK] CLAUDE.md изменён → проверь MemoryBank/MASTER_INDEX.md если поменялась структура"
fi

# Изменён код моделей/генераторов — напоминание про тесты и сверку
if [[ "${FILE_PATH}" =~ core/(models|generators)/.+\.py$ ]]; then
    echo ""
    echo "🧪 [HOOK] Изменён core-код: $(basename "${FILE_PATH}")"
    echo "   → прогони: python tests/all_test.py (НЕ pytest)"
    echo "   → сверь корректность с эталоном NumPy/SciPy"
fi

# Публичный API — обновить README
if [[ "${FILE_PATH}" =~ core/.+/__init__\.py$ ]]; then
    echo ""
    echo "📋 [HOOK] Изменён __init__ (публичный API) → обнови README.md при смене интерфейса"
fi

# Заглавная папка Core — частая ошибка
if [[ "${FILE_PATH}" =~ /Core/ ]]; then
    echo ""
    echo "⚠️  [HOOK] Путь содержит 'Core/' — пакет должен быть строчным 'core/' (Linux ФС регистрозависима)!"
fi

exit 0
