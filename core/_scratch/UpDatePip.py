import subprocess
import sys
import json

# Получаем список устаревших пакетов в JSON формате
result = subprocess.check_output(
	[sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
	text=True
)

packages = [pkg['name'] for pkg in json.loads(result)]

if not packages:
	print("✅ Все пакеты актуальны!")
else:
	print(f"🔄 Обновляем {len(packages)} пакетов: {', '.join(packages)}\n")
	for pkg in packages:
		print(f"  → {pkg}")
		subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', pkg])

	# Обновляем сам pip
	subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'pip'])
	print("\n✅ Готово!")