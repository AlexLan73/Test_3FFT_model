# Ревью: config + data_context + common

Кластер: `core/config/**`, `core/data_context/**`, `common/**` (read-only ревью,
код не правился).

## Находки (severity ↓)

### [HIGH] Молчаливая потеря данных при сериализации `SceneConfig` — `core/data_context/run_workspace.py:61-65`
**Что:** `_scene_to_dict()` сериализует только `scene.emitters` и `scene.thermal`:
```python
def _scene_to_dict(scene: SceneConfig) -> dict[str, Any]:
    return {
        "emitters": [_emitter_to_dict(e) for e in scene.emitters],
        "thermal": {"power": scene.thermal.power},
    }
```
Но `SceneConfig` (`core/config/scene_config.py:94-99`) с P3 несёт ещё 4 поля:
`jammers: JammerFlags`, `barrage_spec`, `comb_spec`, `ham_spec`. Они активно
используются в проде (`core/runtime/scene_server.py:138-221`,
`demo_body_motion_jammers.py:219-220`, `demo_body_motion_multi.py:134-135`,
`tests/test_body_motion_jammers.py`), но при проходе через `config_to_dict`/
`project_config_to_dict` → `to_yaml` → `write_manifest`, а также через
`project_config_from_dict` (round-trip, используется в
`tests/test_body_motion.py:84-85`) — эти 4 поля **бесследно теряются**: ни
записи в `manifest.yaml`, ни восстановления при `from_dict`.
**Почему нарушение:** `config_to_dict`/`RunWorkspace` заявлены как источник
воспроизводимости прогона («Каталог одного прогона… manifest сцены»,
docstring файла) — реальный manifest не отражает реальную сцену, если
включены jammer-флаги/спеки. Это не архитектурная придирка, а баг: OCP тоже
нарушен — добавление полей в `SceneConfig` (P3) не потянуло правку
сериализатора.
**Как исправить:** дополнить `_scene_to_dict`/`_scene_from_dict`-путь (там,
где сейчас `SceneConfig(...)` в `project_config_from_dict`) сериализацией
`jammers` (`dataclasses.asdict(scene.jammers)`) и трёх optional-спек (`None`
→ пропустить/`null`, иначе `_emitter_to_dict`-подобный дамп через тот же
`_EMITTER_TYPES`).

### [MED] `NpyCubeRepository` не использует `pathlib`, methods без type hints — `core/data_context/repository.py:1-34`
**Что:** Весь файл построен на `os`/`os.path.join` (`import os`,
`os.makedirs(root, exist_ok=True)` (стр.23), `os.path.join(...)` (стр.26)),
хотя проектный стиль (`.claude/rules/05-python-style.md`) явно требует
`Pathlib для путей… Минимум os.path.join`. Соседний файл кластера
(`run_workspace.py`) сделан на `Path` — расхождение в рамках одного
кластера. Плюс `NpyCubeRepository.save`/`.load` (стр.28, 33) объявлены без
type hints (`def save(self, name, cube):`), хотя абстрактный `CubeRepository`
двумя строками выше их аннотирует (`def save(self, name: str, cube: np.ndarray) -> str`).
**Почему нарушение:** прямое нарушение явного правила стиля + расхождение
между декларацией контракта (ABC) и его реализацией по типизации.
**Как исправить:** переписать на `Path`, скопировать сигнатуру `save`/`load`
из `CubeRepository` в реализацию.

### [MED] `print()` в библиотечном коде — `common/gpu_context.py:77`
**Что:**
```python
except Exception as exc:
    print(f"[GPUContextManager] dsp_core.ROCmGPUContext(device={device}) failed: {exc}")
```
`common/` — библиотечный код (не `main.py`/demo), правило 05 явно запрещает
`print()` в либе.
**Почему нарушение:** прямое нарушение именованного запрета проекта; помимо
буквы правила — `print` не настраиваем (нет уровня/маршрутизации), тогда как
вызывающий код (`GPUContextManager.get_rocm`) не может ни подавить, ни
перенаправить диагностику ROCm-инициализации.
**Как исправить:** завести module-level `logging.getLogger(__name__)` и
`logger.warning(...)`, либо прокинуть сюда что-то вроде опционального
callback/`warnings.warn`.

### [MED] `TestResult`/`ValidationResult` заявлены как Value Object, но мутабельны — `common/result.py:16-79`
**Что:** Docstring модуля: «Value Objects (GoF) — неизменяемые объекты,
идентифицируются по значению», но оба класса — обычный `@dataclass` (не
`frozen=True`), а `TestResult.add()` (стр.65-68) явно мутирует
`self.validations` через `.append()` и возвращает `self` (fluent, а не новый
объект).
**Почему нарушение:** GoF-паттерн назван неверно/непоследовательно —
реальное поведение (мутация + identity-возврат `self`) это Builder/Pure
Fabrication, а не Value Object. Для читателя, ищущего в коде «неизменяемые
VO» (как остальные конфиги в кластере — все `frozen=True`), это вводит в
заблуждение и создаёт риск: `TestResult`, отданный наружу (например в
`ResultStore`/репортер), можно случайно домутировать после публикации.
**Как исправить:** либо сделать `frozen=True` и `add()` возвращать новый
`TestResult` (реальный VO), либо просто переименовать роль в docstring
(«накопитель результатов теста», не «Value Object»).

### [LOW] Registry для (де)сериализации `EmitterSpec` требует ручной правки в стороннем файле — `core/data_context/run_workspace.py:31-37`, `132-137`
**Что:** `_EMITTER_TYPES` — плоский `dict[str, type[EmitterSpec]]`,
перечисляющий классы вручную. Новый подкласс `EmitterSpec` в
`core/config/scene_config.py` не заработает в `_emitter_from_dict` (тихо
попадёт в `EmitterSpec`-fallback, а `cls(**payload)` упадёт `TypeError` из-за
незнакомых полей — падение громкое, но по месту непонятное), пока кто-то не
вспомнит добавить строку сюда — в другом модуле, другом слое (`data_context`
знает о конкретных сабклассах `config`).
**Почему нарушение:** OCP/DIP: `data_context` (Facade/persistence) обязан
знать весь список конкретных типов сцены вместо того, чтобы типы
регистрировали себя сами (ср. `EmitterFactory` в `generators/factory.py`,
который в CLAUDE.md заявлен как Abstract Factory/Registry — тот же принцип
для генерации, но не для сериализации).
**Как исправить:** не критично сейчас (типов мало, 4 шт.), но при росте
числа спек — завести `EmitterSpec.type_name` classmethod/`__init_subclass__`
регистр, как это сделано в generators/factory.py.

### [LOW] `config_to_dict` диспетчеризует по `isinstance` вместо полиморфизма — `core/data_context/run_workspace.py:44-58`
**Что:** `if isinstance(cfg, ProjectConfig): return project_config_to_dict(cfg)` —
единственная точка, различающая `SimulationConfig`/`ProjectConfig` веткой
`isinstance`. Третий тип конфига потребует новой ветки здесь же.
**Почему нарушение:** мягкое нарушение OCP; на практике оправдано (это
единственный consumer моста между старым и новым конфигом, задокументировано
в docstring как временный мост A5), поэтому не поднимаю выше LOW.
**Как исправить:** не обязательно сейчас; если появится третий тип конфига —
вынести в `Protocol`-метод `to_manifest_dict()` на самих VO.

### [LOW] `load_gpu_config` глотает все исключения без диагностики — `common/gpu_configs.py:36-48`
**Что:** `except Exception: return []` — не bare `except:`, но исключение не
захватывается (`as e`) и никак не логируется; любая ошибка (битый JSON,
права доступа, неожиданная структура) неотличима от «файла нет».
**Почему нарушение:** соответствует букве правила («нужен `except Exception
as e`», а не голый `except:`), но не духу — по факту диагностика так же
теряется, что осложняет отладку выбора GPU-устройства.
**Как исправить:** `except Exception as exc: logger.debug(...); return []`
(или хотя бы `# noqa` с явным комментарием, почему тихо).

### [LOW] Валидаторы (`common/validators/*.py`) без type hints на `actual`/`reference` — `common/validators/base.py:30-33` (и во всех наследниках)
**Что:** `IValidator.validate(self, actual, reference=None, name: str = "")`
— `actual`/`reference` не аннотированы (в docstring указано `scalar / list /
np.ndarray`, но не в сигнатуре). Повторяется во всех 5 конкретных
валидаторах.
**Почему нарушение:** формально нарушает «Type hints везде» из 05-python-style.
Снижена severity: файл явно помечен как vendored 1:1 из DSP-GPU («логика не
менялась»), нет содержательной архитектурной проблемы — только типизация.
**Как исправить:** `actual: ArrayLike, reference: ArrayLike | None = None`
при следующей правке этих файлов (не обязательно отдельным PR).

## Удачно

- `common/validators/*` — эталонный Strategy + Composite + GRASP Creator
  (`ValidatorFactory`): узкие однометодные классы, `_require_reference`
  делает fail-fast явным вместо тихого сравнения с `None`, `CompositeValidator`
  осознанно падает на пустом списке вместо молчаливого PASS (комментарий
  прямо объясняет «почему», а не просто «что»).
- `core/config/*` — конфиги как настоящие frozen VO с валидацией инвариантов
  в `__post_init__` (`ArrayConfig`, `RangeConfig`, `ProjectConfig`); `ProjectConfig`
  честно агрегирует существующие VO по ссылке, не дублирует поля (docstring
  прямо объясняет мотивацию агрегации вместо копирования).
- `core/data_context/message_bus.py` — чистый маленький Observer/Subject:
  `publish()` копирует список подписчиков перед итерацией (`list(...)`,
  стр.41) — защищает от мутации во время notify; `MessageBus` осознанно не
  смешан с `CubeRepository` в `DataContext` (SRP, задокументировано в обоих
  местах).

## Сводка: 1 high / 3 med / 4 low
