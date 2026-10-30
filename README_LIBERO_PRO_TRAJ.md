# Libero-Pro-traj: данные для обучения и экспериментов

Синтетические успешные демонстрации манипуляции роботом Panda в сценах LIBERO,
собранные собственным пайплайном на базе LIBERO-Pro. Датасет предназначен для
imitation learning и кондиционирования гиперсети демонстрациями. Это обучающий
набор с пересечением по задачам и сценам с LIBERO/Pro; сам по себе он не является
независимым held-out бенчмарком.

Датасет: [Kesvill/Libero-Pro-traj](https://huggingface.co/datasets/Kesvill/Libero-Pro-traj).
Датасет содержит **7 628 эпизодов / 219 семантических задач**.

## Версия и основные числа

Для воспроизводимых экспериментов фиксируйте HF revision:

```text
8ada4e5f671e744f000fac08f82bd9f517448500
```

SHA-256 файла `provenance.json` этого релиза:

```text
28fec37c3fa74e50150b0d7e5b592550d0b1e08464a9662c475e582a8ae8fad3
```

Версия формата `v3.0` в `meta/info.json` — не название HF-ветки или тега.
`main` может измениться; для этого релиза используйте SHA выше.

| Показатель | Значение |
|---|---:|
| Формат | LeRobot v3.0 |
| Эпизоды | 7 628 |
| Семантические задачи (`task_program_id`) | 219 |
| Различные тексты инструкций | 152 |
| Родительские BDDL-задачи | 120 |
| Начальные конфигурации (`source_group_id`) | 7 628 |
| Кадры после no-op-фильтра | 1 289 425 |
| Частота управления при записи / FPS видео | 10 Гц / 10 FPS |
| Камеры | agentview и wrist, обе 256 × 256 |
| Длительность: средняя / медиана | 16,90 / 14,0 с |
| Длительность: минимум / максимум | 4,4 / 59,5 с |
| Длительность: p90 / p95 | 31,6 / 35,8 с |
| Эпизодов на задачу | от 1 до 40; у 120 задач по 40 |
| Видео | 15 256 MP4, H.264 |
| Файлы данных / всего файлов пакета | 7 628 Parquet / 22 890 |
| Размер пакета | 27,93 GB, или 26,02 GiB |

На HF дополнительно присутствует служебный `.gitattributes`. README, журналы
сборки и отчёты аудита в HF-пакет не включены; эта документация находится в
репозитории кода.

## Скачать и прочитать

Команды выполняются из корня `fewshot_vla` в подготовленном окружении проекта.
Чтение проверено с Python 3.12, LeRobot 0.5.1 и backend `pyav`.
Используйте отдельную локальную папку для новой ревизии: скачивание поверх
старого набора не гарантирует удаления его лишних файлов.

На общей машине сначала задайте кэши внутри проекта. Переменные действуют
в текущем терминале и его дочерних процессах; конфиги сервера не меняются:

```bash
mkdir -p outputs/cache/{huggingface,datasets,xet,xdg,torch} outputs/tmp
export HF_HOME="$PWD/outputs/cache/huggingface"
export HF_DATASETS_CACHE="$PWD/outputs/cache/datasets"
export HF_XET_CACHE="$PWD/outputs/cache/xet"
export XDG_CACHE_HOME="$PWD/outputs/cache/xdg"
export TORCH_HOME="$PWD/outputs/cache/torch"
export TMPDIR="$PWD/outputs/tmp"
```

```bash
venv/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Kesvill/Libero-Pro-traj",
    repo_type="dataset",
    revision="8ada4e5f671e744f000fac08f82bd9f517448500",
    local_dir=Path("data/libero_pro_traj_8ada4e5"),
    allow_patterns=[
        "data/**", "videos/**", "meta/**",
        "episode_keys.json", "provenance.json",
    ],
)
PY
```

Для публичного датасета HF login не нужен. На общей машине используйте
проектные каталоги кэша; если нужен токен, передавайте его только своему
процессу, без изменения общего login.

```bash
venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import json
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("data/libero_pro_traj_8ada4e5")
info = json.loads((root / "meta/info.json").read_text())
assert (info["fps"], info["total_episodes"], info["total_frames"]) == (
    10, 7628, 1289425,
)
expected = "28fec37c3fa74e50150b0d7e5b592550d0b1e08464a9662c475e582a8ae8fad3"
assert hashlib.sha256((root / "provenance.json").read_bytes()).hexdigest() == expected

ds = LeRobotDataset(
    "Kesvill/Libero-Pro-traj",
    root=root,
    video_backend="pyav",
)
sample = ds[0]
print(ds.num_episodes, len(ds), ds.fps)
for key in (
    "observation.images.image", "observation.images.image2",
    "observation.state", "action",
):
    print(key, tuple(sample[key].shape))
PY
```

Структура данных:

```text
data/chunk-*/file-*.parquet
videos/observation.images.image/chunk-*/file-*.mp4
videos/observation.images.image2/chunk-*/file-*.mp4
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-*/file-*.parquet
episode_keys.json
provenance.json
```

| Ключ | Форма после `LeRobotDataset` | Содержание |
|---|---|---|
| `observation.images.image` | `(3, 256, 256)` | agentview |
| `observation.images.image2` | `(3, 256, 256)` | wrist: `robot0_eye_in_hand` |
| `observation.state` | `(8,)` | EEF xyz, axis-angle RoboSuite, два сустава захвата |
| `action` | `(7,)` | нормированные команды OSC_POSE: delta xyz, delta rotation, захват |

`action` ограничен диапазоном `[-1, 1]`; **−1 открывает захват, +1 закрывает**.
Это относительные команды контроллера, а не абсолютная поза или суставные
углы. Три компоненты поворота относятся к axis-angle, не Euler RPY.
Дополнительное преобразование захвата из `0/1` не нужно.

Изображения уже приведены к конвенции `lerobot-libero`: вертикальный flip
сырых OpenGL-кадров при записи и горизонтальный flip при экспорте согласованы
с обработкой изображений в LeRobot. Повторно отражать или поворачивать
скачанные видео не нужно. Пертурбаций положения/ориентации камер нет;
wrist-камера естественным образом движется вместе с рукой.

## Как определяются задачи и эпизоды

`task_index` связан с текстовой инструкцией и имеет 152 значения. Он не
заменяет 219 семантических идентификаторов. Разные сцены, составы объектов
или последовательности могут иметь одинаковый текст.

Используйте отображение `episode_keys.json["episode_to_key"]`:

```python
from pathlib import Path
import json

root = Path("data/libero_pro_traj_8ada4e5")
keys = json.loads((root / "episode_keys.json").read_text())["episode_to_key"]
task_key = keys["0"]  # pro_<семантический хеш>
```

`task_program_id` учитывает объекты и fixtures, начальные отношения, конечные
цели, заданный порядок промежуточных целей и пространственные/поведенческие
ограничения. ID вычисляется до случайных изменений layout, yaw, цвета,
освещения, динамики и добавления distractors. Эти изменения дают новые
условия исполнения, но не увеличивают число семантических задач.

`provenance.json` содержит поэпизодные `instruction`, `catalog_spec`,
`base_suite`, `base_task`, `goal`, `program`, а для составных задач —
`execution_steps` и `ordered_goals`; также seed, teacher, аугментации,
изменённые параметры модели, хеши BDDL/ассетов и результаты replay.

`source_group_id` характеризует начальную конфигурацию: хешируются имена тел,
их `body_pos` и `qpos` с округлением до `1e-6`. Это не полный хеш мира: масса,
трение, цвета и освещение в него не входят. В этом релизе все 7 628 групп
различны, в том числе внутри каждой задачи.

## Состав задач

Каталог сборщика содержал 250 кандидатов, квота — до 40 успешных эпизодов на
кандидат. В релиз вошли 219 задач; недобор не заполнен неуспешными записями.

| Происхождение определения | Задачи | Эпизоды | Что меняется |
|---|---:|---:|---|
| `native` | 120 | 4 115 | исходные задачи LIBERO, новые исполнения и условия |
| `goal` | 35 | 1 211 | официальные варианты целей LIBERO-Pro из `ood_task.yaml` |
| `object` | 44 | 1 698 | подстановка другого существующего типа упаковки с согласованием BDDL и инструкции |
| `program` | 20 | 604 | наши последовательности с промежуточным размещением |
| **Всего** | **219** | **7 628** | |

В `program`: 16 задач `via` / 501 эпизод и 4 `round_trip` / 103 эпизода.
`via` требует промежуточного размещения перед основной целью;
`round_trip` — основной цели, промежуточного размещения и возврата к основной
цели. Одного совпадения конечного состояния недостаточно для их успеха.
Поле `program="native"` у остальных записей не означает `kind=native`:
варианты `goal` и `object` тоже могут иметь это значение.

| Базовый suite | Родительские BDDL | Задачи релиза с этим источником | Эпизоды |
|---|---:|---:|---:|
| `libero_goal` | 10 | 35 | 1 151 |
| `libero_object` | 10 | 40 | 1 595 |
| `libero_spatial` | 10 | 18 | 712 |
| `libero_10` | 9 | 17 | 493 |
| `libero_90` | 81 | 109 | 3 677 |
| **Всего** | **120** | **219** | **7 628** |

По механизмам исполнения:

| Навык сборщика | Задачи | Эпизоды |
|---|---:|---:|
| Pick-and-place | 145 | 5 267 |
| Последовательности / несколько целей | 48 | 1 449 |
| Открытие ящиков | 6 | 196 |
| Закрытие ящиков | 4 | 118 |
| Открыть ящик и положить предмет | 9 | 323 |
| Толкание без подъёма/захвата | 2 | 75 |
| Управление плитой | 5 | 200 |

Это классификация исполнения, а не семь непересекающихся человеческих
понятий навыка: например, составная программа может включать pick-and-place.

В определениях присутствуют 26 типов подвижных объектов: миски, тарелки,
бутылки, кружки, книги, moka pot, сковорода, корзины/подносы, ramekin, cookies
и упаковки продуктов. Заменяются существующие LIBERO-категории
`alphabet_soup`, `butter`, `cream_cheese`, `chocolate_pudding`, `tomato_sauce`,
`ketchup`, `salad_dressing`, `bbq_sauce`, `milk`, `orange_juice`;
новые 3D-меши этим релизом не вводятся. Fixtures включают плиту,
микроволновку, шкафы, полку, wine rack и desk caddy.

| Рабочая область исходной сцены | Задачи | Эпизоды |
|---|---:|---:|
| Общий стол (`table`) | 53 | 1 863 |
| Пол (`floor`) | 40 | 1 595 |
| Кухонный стол | 58 | 1 869 |
| Стол в гостиной | 56 | 2 074 |
| Письменный стол | 12 | 227 |

В именах исходных задач представлены 19 сцен `KITCHEN_SCENE*`,
`LIVING_ROOM_SCENE*`, `STUDY_SCENE*`, дополнительно используются общие
tabletop/floor постановки. 120 родительских задач не означают 120 независимых
трёхмерных окружений. Полный поимённый каталог с числом эпизодов приведён ниже.

## Какие пертурбации использованы

Таблица считает объявленные оси в `augmentations`; реальные значения для
каждого эпизода находятся в `model_overrides`, `initial_yaw_radians`,
`distractor_objects` и параметрах сцены.

| Ось | Диапазон / способ | Задачи | Эпизоды |
|---|---|---:|---:|
| Layout | смещение областей размещения до ±2,5 см по x/y | 201 | 6 916 |
| Yaw | случайный поворот объектов до ±π/2 | 219 | 7 628 |
| Dynamics | масса и инерция ×0,7–1,3; скользящее трение ×0,7–1,3 | 219 | 7 628 |
| Distractors | до двух дополнительных упаковок | 201 | 6 916 |
| Colors | RGB визуальных геометрий объектов 0,08–0,9 | 153 | 5 301 |
| Lighting | diffuse ×0,6–1,3, ambient 0,03–0,25 | 219 | 7 628 |
| Камеры | поза камер не аугментируется | 0 | 0 |

Фактически два distractor-объекта есть в 6 416 эпизодах, один — в 450,
ни одного — в 762: включённая ось не гарантирует свободного места для вставки.
Для spatial-задач layout, colors и distractors отключены, чтобы сохранять
значение пространственной инструкции. Цвета также не меняются, если текст
содержит `black/white/red/yellow/green/blue`.

Освещение может менять вид всей сцены, включая робота и мебель, хотя
перекраска напрямую применяется только к объектам. Для цветозависимых
инструкций новые случайные источники света используют нейтральные RGB;
при восстановлении исходных сцен базовое освещение дополнительно
возвращено в 1 447 эпизодах. Поэтому наличие `lighting` в записи не означает,
что во всех эпизодах свет обязательно отличается от исходного.

Маршруты `left/right/high` распределены как 2 546 / 2 606 / 2 476 эпизодов.
Это метки стратегии/ориентира сбора, а не изменение камеры и не новые задачи;
у X-VLA фактический путь выбирает политика.

## Как собирались траектории

1. Исходные сцены и ассеты взяты из LIBERO-Pro, commit
   `eafdb809426b13153aa1e4c42d6601844217dfec`. Собственный генератор формирует
   согласованные BDDL, инструкции, объекты и, где нужно, порядок подцелей.
2. Для сцены создаётся начальная конфигурация, применяются допустимые
   аугментации. Нестабильные, уже решённые и некорректно отрендеренные
   состояния отклоняются до принятия эпизода.
3. Исполнитель управляет настоящей симуляцией через `env.step` с
   **`control_freq=10`**, записывая новые наблюдения и реальные относительные
   OSC-действия. Видео кодируется в 10 FPS.
4. Успешное исполнение проходит отдельный action replay из сохранённой
   модели/начального состояния, затем экспортируется в LeRobot v3.
5. После объединения применяется no-op-фильтр, проверяются все числовые
   строки, сохранённые кадры, их PTS и соответствие источникам.

В финальном релизе:

| Исполнитель в `teacher` | Эпизоды | Как получены действия |
|---|---:|---|
| `scripted_privileged_v2` | 2 174 | собственный скриптовый исполнитель с доступом к состоянию симулятора |
| `reference` | 5 032 | замкнутое исполнение на 10 Гц по абсолютным OSC-ориентирам исходной успешной траектории |
| `xvla` | 422 | предсказания X-VLA, преобразованные в OSC-команды и выполненные в симуляции |

Для `reference` использовались начальные сцены и ориентиры исходных успешных
20-Гц записей. Наблюдения и действия получены при исполнении в симуляции на
10 Гц. Связь с исходными записями сохраняется в `recollection_source` и полях
`reference_*`. При объединении с этими источниками связанные эпизоды нужно
держать в одной группе при разбиении на train/test и support/query.

X-VLA: [2toINF/X-VLA-Libero](https://huggingface.co/2toINF/X-VLA-Libero),
checkpoint revision `129e71460678b7236cee6fc9707f09d9fa0c3590`,
код `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`, 10 flow-шагов,
`float32`, absolute EE6D на выходе модели. Это учитель сбора, а не модель,
обученная в рамках выпуска датасета.

Готовые демонстрации LIBERO-Gen в этот набор не включались. Исполнения
синтетические; человеческая телеперация и оптимальность движений не заявляются.
Среда сбора: MuJoCo 3.2.7, RoboSuite 1.4.1, NumPy 2.2.6.

### Контроль качества и no-op

При принятии проверяются достижение цели, порядок подцелей, устойчивый успех,
освобождение захвата и валидность изображений. Для push-задач дополнительно
проверяются контакт и отсутствие переноса предмета в захвате. Лимит записи —
600 шагов; нецелевые неподвижные участки длиннее 1 с отклоняются, при этом
необходимые фазы закрытия/открытия захвата и удержания успеха учитываются
отдельно. У X-VLA также ограничены повторные захваты и отсутствие прогресса.

No-op-фильтр следует критерию
[OpenVLA](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/regenerate_libero_dataset.py):
оставляет действие, если `norm(action[:6]) >= 1e-4` либо меняется команда
захвата относительно последнего сохранённого действия. Первое действие
сохраняется только при ненулевом движении. Удалено **13 103 из 1 302 528
кадров**; ни один эпизод целиком не удалён.

Это фильтрация команд, а не детектор визуального движения. Ненулевая команда
при почти неподвижной руке может остаться; фаза работы захвата также может
выглядеть как короткая остановка. Фильтр не обещает движение на каждом кадре.

После удаления строк `timestamp` и видео PTS заново идут с шагом 0,1 с.
`source_step_index.indices` сохраняет номера исходных шагов. Длительности
в этом README измерены после фильтрации.

**Физический replay проверял исходные записанные исполнения. После no-op
фильтрации последовательность в симуляции заново не исполнялась:** в метаданных
`filtered_sequence_replayed=false`. Для отфильтрованного набора проверены
точное совпадение выбранных действий/состояний с источниками, все сохранённые
RGB-кадры и PTS, SHA-256 всех 15 256 выходных видео. Повторное кодирование
проверено на совпадение декодированных RGB. Числовых дубликатов траекторий
внутри релиза не найдено.

Выгрузка содержит данные для обучения и метаданные происхождения, но не полный
комплект MuJoCo XML/init/raw HDF5 для воспроизведения action replay.

## Пересечение с LIBERO и LIBERO-Pro

### Исходные сцены и задачи

Сравнены родительские BDDL всех пяти suites: LIBERO-Pro
`eafdb809426b13153aa1e4c42d6601844217dfec` и original LIBERO
`8f1084e3132a39270c3a13ebe37270a43ece2a01`. **130 из 130 файлов побайтово
совпали по SHA-256**, включая все 120 родительских BDDL, представленных в релизе.
Поэтому все задачи имеют общую сценовую/объектную основу с original LIBERO.

Из 120 `native` определений 117 до аугментаций сохраняют семантическое
определение исходного BDDL,
а в трёх задачах включения плиты генератор добавляет явное начальное `Turnoff`.
У этих трёх цель, инструкция, объекты и исходная сцена остаются прежними.
35 `goal` определений взяты из официального Pro YAML. Остальные 64 — наши
подстановки объектов и составные программы на тех же сценах; отсутствие
совпадений с любым возможным вариантом официального Pro для них не доказано.

Это собственный collection pipeline, а не полный официальный sweep
`use_task/use_swap/use_object/use_language/use_environment`. В частности,
названия осей и диапазоны наших аугментаций не означают точного совпадения
с оценочной матрицей Pro. Обучение на соответствующих Pro-целях исключает
интерпретацию их последующего тестирования как переноса на невиданные цели.

Поимённые задачи и хеши позволяют проверить происхождение, но не доказывают
отсутствие демонстраций/задач в предобучении базовой модели или учителя.
Побайтовое сравнение с полными официальными LIBERO human-demo датасетами
в этом аудите не проводилось. Новые записи и отсутствие внутренних дубликатов
не означают отсутствия общих сцен, целей и начальных условий с другими наборами.

### Пересечение с текущим LIBERO-90 split проекта

Ниже — точное сопоставление по `base_suite == libero_90` и
`base_task == bddl_name` из [configs/libero90_split.json](https://github.com/AdilHAA/fewshot_vla/blob/39d943c403f1cae88ee7358db47cb02bc6872988/configs/libero90_split.json)
на commit `39d943c403f1cae88ee7358db47cb02bc6872988`; текст инструкций
для сопоставления не использован.

| Категория релиза | Родители из configured train-40 | Родители из configured eval-50 |
|---|---:|---:|
| `native` | 37 задач / 1 234 эпизода | 44 задачи / 1 440 эпизодов |
| Производные `object/program` | 14 задач / 531 эпизод | 14 задач / 472 эпизода |
| **Все с источником LIBERO-90** | **51 задача / 1 765 эпизодов** | **58 задач / 1 912 эпизодов** |

Таким образом, обучение гиперсети на всех 7 628 эпизодах включает 44 исходные
задачи из configured `eval-50`. После такого обучения называть их невиданными
для всей системы нельзя. Проверять нужно отдельно экспозицию frozen base,
гиперсети, банка демонстраций и данных выбора гиперпараметров.

Task 51 (`LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket`)
есть в датасете: `pro_b16f6ba81b3127e24e9a`, 37 эпизодов, и производная
`pro_7ac3d77e3a427e520d60`, 37 эпизодов. [LIBERO-90 model card](MODEL_CARD.md) сообщает,
что эта задача отсутствовала в данных дообучения базы. Таблица выше сравнивает
конфигурацию, а не заменяет аудит `train_config` и episode lists конкретного
чекпоинта. Она также не является полным семантическим поиском пересечений
между разными suites.

## Использование в экспериментах гиперсети

### Данные, splits и support/query

Весь релиз помечен `split="train"`. Готовых независимых train/validation/test
сплитов в нём нет. `source_group_id` отделяет начальные конфигурации;
disjoint задачи автоматически не подразумеваются.

Для проверки новых задач разделяйте по `task_program_id`; для проверки новых
начальных условий внутри задачи — по `source_group_id`. Support и query
должны следовать объявленному протоколу. Не используйте эпизоды query для
градиентного обучения, выбора демонстраций или настройки гиперпараметров,
если они объявлены held-out. Старые `train_episodes.json`/`task_id` от
LIBERO-90 не являются списками эпизодов этого датасета.

Для strict cross/leave-one-out доступны **215 задач / 7 624 эпизода**.
Четыре задачи имеют единственный эпизод:

```text
pro_0b1c687e3ddc97a406c8
pro_30e28c5448dea8cba910
pro_e740732c7feef94db754
pro_f2e96fa29e3b67043612
```

Текущий selector при отсутствии другого эпизода откатывается на self даже
при `p_self=0`. Для strict cross эти четыре задачи нужно явно исключить
из соответствующего экспериментального subset и сообщить об этом; не
исключайте их молча из общего релиза. Внутри именно этой версии разные
эпизоды имеют разные source groups, но после объединения наборов это свойство
надо проверить снова.

Если выбран именно strict-cross протокол, список подходящих эпизодов можно
сохранить отдельно. Этот subset исключает четыре singleton-задачи, но не
создаёт train/test split по задачам:

```bash
venv/bin/python - <<'PY'
from collections import defaultdict
from pathlib import Path
import json

root = Path("data/libero_pro_traj_8ada4e5")
episodes = json.loads((root / "provenance.json").read_text())["episodes"]
groups = defaultdict(set)
for episode in episodes:
    groups[episode["task_program_id"]].add(episode["source_group_id"])
eligible = {task for task, values in groups.items() if len(values) >= 2}
selected = sorted(
    episode["episode_index"] for episode in episodes
    if episode["task_program_id"] in eligible
)
assert (len(eligible), len(selected)) == (215, 7624)
path = Path("outputs/splits/libero_pro_8ada4e5_cross_episodes.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(selected) + "\n")
print(path)
PY
```

### Построить новый cache

После смены релиза создайте отдельный cache. В текущем `CacheHeader` нет
fingerprint/revision/FPS датасета, поэтому старый cache может пройти проверку
формата и при этом содержать другие эпизоды. Передавайте именно поставляемый
`episode_keys.json`; `--legacy_keys` и ключи по тексту здесь не подходят.

```bash
venv/bin/python scripts/build_xpair_cache.py \
  --repo_id Kesvill/Libero-Pro-traj \
  --root data/libero_pro_traj_8ada4e5 \
  --video_backend pyav \
  --keys data/libero_pro_traj_8ada4e5/episode_keys.json \
  --encoder dino \
  --workers 8 \
  --out outputs/xpair_cache/libero_pro_traj_8ada4e5_dino
```

Это команда построения эмбеддингов, а не запуск обучения. Для V-JEPA2/Qwen
используйте существующий рецепт выбранного arm, сохраняя те же dataset root,
semantic keys и согласованный список эпизодов. У cache builders `--episodes`
принимает путь к JSON-массиву индексов эпизодов.

### Подключить к выбранному рецепту обучения

Параметры данных для существующего согласованного запуска:

```text
--dataset.repo_id=Kesvill/Libero-Pro-traj
--dataset.root=data/libero_pro_traj_8ada4e5
--dataset.video_backend=pyav
--policy.hn_xpair_cache_path=outputs/xpair_cache/libero_pro_traj_8ada4e5_dino
```

У `scripts/train.sh` им соответствуют `DATASET`, `DATASET_ROOT`,
`VIDEO_BACKEND`, `XPAIR_CACHE`. Для выбранного JSON-списка эпизодов есть
`EPISODES_FILE`; `EPISODES` принимает сам JSON-массив, не имя файла.
Не задавайте оба одновременно. Подмножества cache/support/query должны
соответствовать протоколу, а не только списку BC-эпизодов.

Менять одновременно набор данных, arm, базовый чекпоинт, `p_self`, число
демонстраций, бюджет, train/eval split и нормализацию нельзя, если требуется
сравнение влияния датасета. В стандартном запуске `HN_PIN_BASE_STATS=1`
сохраняет state/action normalizer frozen base; статистики нового набора не
подменяют его автоматически. Для Qwen временной разметке соответствует
`hn_trunk_fps=10`.

При смешивании с другими наборами проверяйте физическую частоту управления,
действия, gripper, конвенцию изображений и semantic keys. Одинаковая метка
`fps=10` не доказывает одинаковой физической временной шкалы.

### Что поддерживает опубликованный код

Совместимость проверена по `fewshot_vla/master` на commit
`39d943c403f1cae88ee7358db47cb02bc6872988`:

- скачивание, чтение LeRobot и построение cache с `--root`/`--keys` доступны;
- dataset/cache параметры обучения и `EPISODES_FILE` доступны;
- автоматическое исключение одинаковых `source_group_id` после объединения
  наборов отсутствует: selector исключает только тот же episode;
- simulator hook для физического `control_freq=10` и сопоставление `pro_*`
  с task keys оценочных сред в этой версии ещё не опубликованы.

Поэтому `LIBERO_CONTROL_FPS=10` / `HN_TASK_KEY_MODE=semantic` сами по себе
не настраивают eval этого commit. Одного `--env.fps=10` также недостаточно
для изменения физики. До симуляторного эксперимента нужны соответствующий
frequency hook, явное сопоставление семантических задач и правильные BDDL/init;
для `via/round_trip` — ещё и проверка порядка подцелей. Stock LIBERO success
по конечному goal не заменяет эту проверку.

При публикации результатов фиксируйте HF SHA, checksum provenance, код и
чекпоинт базы, списки train/support/query задач и эпизодов, cache-конфиг,
нормализацию, бюджет и физическую частоту eval. Success rate на виденных
задачах с новыми init, на новых целях и на новых сценах — разные эксперименты.

## Источники

- [Original LIBERO, закреплённые BDDL](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/bddl_files).
- [LIBERO-Pro, закреплённый код](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec)
  и [официальные варианты целей](https://github.com/Zxy-MLlab/LIBERO-PRO/blob/eafdb809426b13153aa1e4c42d6601844217dfec/libero_ood/ood_task.yaml).
- [Релиз данных на HF](https://huggingface.co/datasets/Kesvill/Libero-Pro-traj/tree/8ada4e5f671e744f000fac08f82bd9f517448500):
  `meta/info.json`, `meta/episodes`, `episode_keys.json`, `provenance.json`.
- [X-VLA LIBERO checkpoint](https://huggingface.co/2toINF/X-VLA-Libero/tree/129e71460678b7236cee6fc9707f09d9fa0c3590).

## Полный каталог задач

Таблицы ниже получены из `provenance.json` закреплённого релиза. Количество
эпизодов включает все teachers и считается после no-op-фильтрации.
`suite` указывает источник BDDL, а не отдельный экспериментальный split.

<details>
<summary>Исходные задачи LIBERO — 120 задач, 4115 эпизодов</summary>

| task_program_id | Источник BDDL: suite / stem | Эпизоды | Инструкция |
|---|---|---:|---|
| `pro_15e2689e08d21ca8fd1c` | `libero_10 / KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 23 | turn on the stove and put the moka pot on it |
| `pro_72a11f1c6ac3bb16ca43` | `libero_10 / KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 31 | put the black bowl in the bottom drawer of the cabinet and close it |
| `pro_7f2a00972ac71fbb2f69` | `libero_10 / KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it` | 7 | put the yellow and white mug in the microwave and close it |
| `pro_5a4a89feaebaaca89b39` | `libero_10 / KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 7 | put both moka pots on the stove |
| `pro_a6ef1d7358bda1c7f901` | `libero_10 / LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 40 | put both the alphabet soup and the cream cheese box in the basket |
| `pro_40080149d70f671e5f1e` | `libero_10 / LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 40 | put both the alphabet soup and the tomato sauce in the basket |
| `pro_f12ecb0a148b42be490d` | `libero_10 / LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 40 | put both the cream cheese box and the butter in the basket |
| `pro_60ac9fc1b81b23aaeef9` | `libero_10 / LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate` | 40 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| `pro_f44b6088454e648e8cd1` | `libero_10 / LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 24 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| `pro_1ae954e0aef3b65b88cb` | `libero_90 / KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet` | 40 | close the top drawer of the cabinet |
| `pro_dec589a401f0a4b97ada` | `libero_90 / KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it` | 40 | close the top drawer of the cabinet and put the black bowl on top of it |
| `pro_7645ec5b9cee5c585af4` | `libero_90 / KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` | 40 | put the black bowl in the top drawer of the cabinet |
| `pro_6d8132c1467c5b1a420f` | `libero_90 / KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it` | 40 | put the butter at the back in the top drawer of the cabinet and close it |
| `pro_a146070a4ad825d5a632` | `libero_90 / KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it` | 40 | put the butter at the front in the top drawer of the cabinet and close it |
| `pro_7487e7b8b3c756506666` | `libero_90 / KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it` | 40 | put the chocolate pudding in the top drawer of the cabinet and close it |
| `pro_809fba978bb5d731df74` | `libero_90 / KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet` | 35 | open the bottom drawer of the cabinet |
| `pro_bf690e955f4ffd9dfb8f` | `libero_90 / KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet` | 40 | open the top drawer of the cabinet |
| `pro_2aa727abab26c6ca8b0c` | `libero_90 / KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it` | 37 | open the top drawer of the cabinet and put the bowl in it |
| `pro_9ce9f9801e5023077ab7` | `libero_90 / KITCHEN_SCENE1_put_the_black_bowl_on_the_plate` | 40 | put the black bowl on the plate |
| `pro_948087ffcf5635384e24` | `libero_90 / KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet` | 40 | put the black bowl on top of the cabinet |
| `pro_60568b3a6d50d97f8468` | `libero_90 / KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet` | 40 | open the top drawer of the cabinet |
| `pro_38e5bd015748387308c2` | `libero_90 / KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate` | 40 | put the black bowl at the back on the plate |
| `pro_f69224b83798e0a3a918` | `libero_90 / KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate` | 40 | put the black bowl at the front on the plate |
| `pro_e1409ca9e4a88bbb047d` | `libero_90 / KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate` | 40 | put the black bowl in the middle on the plate |
| `pro_fca182d3a1cbebf22248` | `libero_90 / KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet` | 40 | put the middle black bowl on top of the cabinet |
| `pro_82a5d7bbe08ffc6cf325` | `libero_90 / KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle` | 40 | stack the black bowl at the front on the black bowl in the middle |
| `pro_8a613057ff8d1a2144f8` | `libero_90 / KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl` | 40 | stack the black bowl in the middle on the black bowl at the front |
| `pro_00f38571b14866e0aa22` | `libero_90 / KITCHEN_SCENE3_put_the_frying_pan_on_the_stove` | 20 | put the frying pan on the stove |
| `pro_79e9f7c702129c151852` | `libero_90 / KITCHEN_SCENE3_put_the_moka_pot_on_the_stove` | 27 | put the moka pot on the stove |
| `pro_b513045457483a7a9245` | `libero_90 / KITCHEN_SCENE3_turn_on_the_stove` | 40 | turn on the stove |
| `pro_d9102f3bf7b6331fd0da` | `libero_90 / KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it` | 17 | turn on the stove and put the frying pan on it |
| `pro_c8f7d461444f7139439b` | `libero_90 / KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet` | 33 | close the bottom drawer of the cabinet |
| `pro_ecca0934087da72dbcda` | `libero_90 / KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer` | 38 | close the bottom drawer of the cabinet and open the top drawer |
| `pro_3c969c9ee8019bee0afb` | `libero_90 / KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet` | 39 | put the black bowl in the bottom drawer of the cabinet |
| `pro_075481308a0dcb1f7cc6` | `libero_90 / KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet` | 40 | put the black bowl on top of the cabinet |
| `pro_c5c21b52d5a80fea1c7a` | `libero_90 / KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet` | 39 | put the wine bottle in the bottom drawer of the cabinet |
| `pro_7028279cbbc51a405b39` | `libero_90 / KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack` | 35 | put the wine bottle on the wine rack |
| `pro_4a8ff6f1e61fff3789d7` | `libero_90 / KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet` | 38 | close the top drawer of the cabinet |
| `pro_39ee48af8dac0d9fd6eb` | `libero_90 / KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` | 40 | put the black bowl in the top drawer of the cabinet |
| `pro_42bf9df9521a97a2a54e` | `libero_90 / KITCHEN_SCENE5_put_the_black_bowl_on_the_plate` | 40 | put the black bowl on the plate |
| `pro_975759be0aa640f0a4e0` | `libero_90 / KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet` | 39 | put the black bowl on top of the cabinet |
| `pro_d9ae7ee94cf751537145` | `libero_90 / KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet` | 36 | put the ketchup in the top drawer of the cabinet |
| `pro_ab7d6922e0e70b90c5a4` | `libero_90 / KITCHEN_SCENE6_close_the_microwave` | 7 | close the microwave |
| `pro_faa13f7602edb7ff25e9` | `libero_90 / KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug` | 33 | put the yellow and white mug to the front of the white mug |
| `pro_58263de251a65131ec31` | `libero_90 / KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove` | 37 | put the right moka pot on the stove |
| `pro_26ffd6378bf3bdb41505` | `libero_90 / KITCHEN_SCENE8_turn_off_the_stove` | 40 | turn off the stove |
| `pro_393f82d888efa9fec89f` | `libero_90 / KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet` | 10 | put the frying pan on top of the cabinet |
| `pro_47cf11a148b2750deba1` | `libero_90 / KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf` | 2 | put the frying pan under the cabinet shelf |
| `pro_74e4d3d262d743d1b19d` | `libero_90 / KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet` | 40 | put the white bowl on top of the cabinet |
| `pro_15f668f7e2238088fd8c` | `libero_90 / KITCHEN_SCENE9_turn_on_the_stove` | 40 | turn on the stove |
| `pro_18109e640c181ca1af6f` | `libero_90 / KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it` | 12 | turn on the stove and put the frying pan on it |
| `pro_f7f3c431aac5b28ad591` | `libero_90 / LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket` | 39 | pick up the alphabet soup and put it in the basket |
| `pro_4f9a5aaf7d35d5208bbb` | `libero_90 / LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket` | 36 | pick up the cream cheese box and put it in the basket |
| `pro_c5a98c4da17c16e5a86d` | `libero_90 / LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket` | 39 | pick up the ketchup and put it in the basket |
| `pro_59af3d3de9cb4abbcffd` | `libero_90 / LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket` | 40 | pick up the tomato sauce and put it in the basket |
| `pro_99aa995fc5a7f32f46f3` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket` | 39 | pick up the alphabet soup and put it in the basket |
| `pro_b16f6ba81b3127e24e9a` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket` | 37 | pick up the butter and put it in the basket |
| `pro_7245a0414a7577306351` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket` | 40 | pick up the milk and put it in the basket |
| `pro_cbeb16d189f5ec78c7eb` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket` | 40 | pick up the orange juice and put it in the basket |
| `pro_93f14cc9965401820e15` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket` | 39 | pick up the tomato sauce and put it in the basket |
| `pro_75a0196b601aa261d711` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray` | 40 | pick up the alphabet soup and put it in the tray |
| `pro_4d387bd6ec91f190c117` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray` | 40 | pick up the butter and put it in the tray |
| `pro_3b09f6df78b9469bea2d` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray` | 40 | pick up the cream cheese and put it in the tray |
| `pro_47c70ed952105fe5f47b` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray` | 40 | pick up the ketchup and put it in the tray |
| `pro_b871432163dc74713166` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray` | 39 | pick up the tomato sauce and put it in the tray |
| `pro_4effbdd2b9e31c6ad17a` | `libero_90 / LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray` | 40 | pick up the black bowl on the left and put it in the tray |
| `pro_a43d5024e4dd7ffc2f0d` | `libero_90 / LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray` | 40 | pick up the chocolate pudding and put it in the tray |
| `pro_e1e04d60a3d2206af671` | `libero_90 / LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray` | 37 | pick up the salad dressing and put it in the tray |
| `pro_8d251d710bbad56c6c9b` | `libero_90 / LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray` | 21 | stack the left bowl on the right bowl and place them in the tray |
| `pro_7a4032af90e6003484a4` | `libero_90 / LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray` | 30 | stack the right bowl on the left bowl and place them in the tray |
| `pro_9496c65ec88c0c6279b3` | `libero_90 / LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate` | 23 | put the red mug on the left plate |
| `pro_83d68a119f3d79d06d60` | `libero_90 / LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate` | 30 | put the red mug on the right plate |
| `pro_1cb8ca1204c4d76b0a82` | `libero_90 / LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate` | 39 | put the white mug on the left plate |
| `pro_603b73329efc2e8e52ed` | `libero_90 / LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate` | 37 | put the yellow and white mug on the right plate |
| `pro_5dfcbb070f7dd79a3b92` | `libero_90 / LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate` | 39 | put the chocolate pudding to the left of the plate |
| `pro_1f967207a06086e4ac19` | `libero_90 / LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate` | 40 | put the chocolate pudding to the right of the plate |
| `pro_46fda241987f73be634d` | `libero_90 / LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate` | 32 | put the red mug on the plate |
| `pro_a65d6364b50c1deb9fb9` | `libero_90 / LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate` | 37 | put the white mug on the plate |
| `pro_f2e96fa29e3b67043612` | `libero_90 / STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy` | 1 | pick up the book and place it in the front compartment of the caddy |
| `pro_727f601014fb3a8e7f45` | `libero_90 / STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy` | 28 | pick up the book and place it in the left compartment of the caddy |
| `pro_57ede067c5f0ec611928` | `libero_90 / STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy` | 7 | pick up the book and place it in the right compartment of the caddy |
| `pro_1e05eea2522531b09ee4` | `libero_90 / STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy` | 36 | pick up the yellow and white mug and place it to the right of the caddy |
| `pro_96a7d95b5456a605dbe4` | `libero_90 / STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy` | 28 | pick up the book and place it in the left compartment of the caddy |
| `pro_4b863c25af36de755166` | `libero_90 / STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy` | 8 | pick up the book and place it in the right compartment of the caddy |
| `pro_8e7e6cf27f681e712408` | `libero_90 / STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy` | 28 | pick up the book and place it in the left compartment of the caddy |
| `pro_d866d2bb80baed623bbc` | `libero_90 / STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy` | 17 | pick up the book and place it in the right compartment of the caddy |
| `pro_fdbc1790f134746a2dea` | `libero_90 / STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy` | 37 | pick up the white mug and place it to the right compartment of the caddy |
| `pro_2a2244089db434a8a808` | `libero_90 / STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf` | 8 | pick up the book in the middle and place it on the cabinet shelf |
| `pro_a58b1b050ab2ff7df71c` | `libero_90 / STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf` | 23 | pick up the book on the left and place it on top of the shelf |
| `pro_cb48c46e63fa618d7f83` | `libero_90 / STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf` | 6 | pick up the book on the right and place it on the cabinet shelf |
| `pro_347142a4573df3f24667` | `libero_goal / open_the_middle_drawer_of_the_cabinet` | 40 | Open the middle layer of the drawer |
| `pro_23386c1332756e942e15` | `libero_goal / open_the_top_drawer_and_put_the_bowl_inside` | 34 | Open the top layer of the drawer and put the bowl inside |
| `pro_0dae3c68c60c3d823b8a` | `libero_goal / push_the_plate_to_the_front_of_the_stove` | 35 | Push the plate to the front of the stove |
| `pro_c4267cd13f06d140c399` | `libero_goal / put_the_bowl_on_the_plate` | 40 | Put the bowl on the plate |
| `pro_b2b7318f25a828ce12eb` | `libero_goal / put_the_bowl_on_the_stove` | 40 | Put the bowl on the stove |
| `pro_c0af7ad66d9094a7c2cb` | `libero_goal / put_the_bowl_on_top_of_the_cabinet` | 40 | Put the bowl on the top of the drawer |
| `pro_850cbb226bff1365346a` | `libero_goal / put_the_cream_cheese_in_the_bowl` | 40 | Put the cream cheese on the bowl |
| `pro_0a9e51c4b75c3b483b3c` | `libero_goal / put_the_wine_bottle_on_the_rack` | 40 | Put the wine bottle on the rack |
| `pro_2467efac05ce242aad29` | `libero_goal / put_the_wine_bottle_on_top_of_the_cabinet` | 40 | Put the wine bottle on the top of the drawer |
| `pro_f36ae3a1e51503675e3b` | `libero_goal / turn_on_the_stove` | 40 | Turn on the stove |
| `pro_aee238f93eddffbd74c2` | `libero_object / pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 40 | Pick the alphabet soup and place it in the basket |
| `pro_e0a464b0e413190e78e4` | `libero_object / pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 40 | Pick the bbq sauce and place it in the basket |
| `pro_b5f3f9776931509f6a72` | `libero_object / pick_up_the_butter_and_place_it_in_the_basket` | 40 | Pick the butter and place it in the basket |
| `pro_c22fac391c2cabae9adf` | `libero_object / pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 40 | Pick the chocolate pudding and place it in the basket |
| `pro_94df348299b57b38f24a` | `libero_object / pick_up_the_cream_cheese_and_place_it_in_the_basket` | 40 | Pick the cream cheese and place it in the basket |
| `pro_274dce1f8997114b6c52` | `libero_object / pick_up_the_ketchup_and_place_it_in_the_basket` | 40 | Pick the ketchup and place it in the basket |
| `pro_e6c0e874096dce5683c1` | `libero_object / pick_up_the_milk_and_place_it_in_the_basket` | 40 | Pick the milk and place it in the basket |
| `pro_c60f30b9c0922df39b1d` | `libero_object / pick_up_the_orange_juice_and_place_it_in_the_basket` | 40 | Pick the orange juice and place it in the basket |
| `pro_990ffe0953f1190c6258` | `libero_object / pick_up_the_salad_dressing_and_place_it_in_the_basket` | 40 | Pick the salad dressing and place it in the basket |
| `pro_4304ec53f534a7704ce4` | `libero_object / pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 40 | Pick the tomato sauce and place it in the basket |
| `pro_37d0a77fb498633bc6e0` | `libero_spatial / pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 40 | Pick the akita black bowl between the plate and the ramekin and place it on the plate |
| `pro_330a047620f324cc23c6` | `libero_spatial / pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 40 | Pick the akita black bowl from table center and place it on the plate |
| `pro_9496ced4e5763d0b676f` | `libero_spatial / pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 40 | Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate |
| `pro_b4458607af0770af69b3` | `libero_spatial / pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 40 | Pick the akita black bowl next to the cookies box and place it on the plate |
| `pro_8e24e8fcfb82968ba5d6` | `libero_spatial / pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 40 | Pick the akita black bowl next to the plate and place it on the plate |
| `pro_343416271652438fdd66` | `libero_spatial / pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 40 | Pick the akita black bowl next to the ramekin and place it on the plate |
| `pro_3345302922262bb321c1` | `libero_spatial / pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate` | 40 | Pick the akita black bowl on the cookies box and place it on the plate |
| `pro_5c327df195724968d998` | `libero_spatial / pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 40 | Pick the akita black bowl on the ramekin and place it on the plate |
| `pro_cf25d7c17db8617f5c0b` | `libero_spatial / pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate` | 40 | Pick the akita black bowl on the stove and place it on the plate |
| `pro_945f075d01943ab50328` | `libero_spatial / pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 40 | Pick the akita black bowl on the wooden cabinet and place it on the plate |

</details>

<details>
<summary>Официальные варианты целей LIBERO-Pro — 35 задач, 1211 эпизодов</summary>

| task_program_id | Источник BDDL: suite / stem | Эпизоды | Инструкция |
|---|---|---:|---|
| `pro_7e61ab980655764090f1` | `libero_10 / KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it` | 2 | put the bottle in the bottom drawer of the cabinet and close it |
| `pro_8e5d36bf61379205cf98` | `libero_10 / LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 36 | put both the alphabet soup and the cream cheese in the basket |
| `pro_be4e7fb983869d980bcd` | `libero_10 / LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 39 | put both the cream cheese and the tomato sauce in the basket |
| `pro_918cc0fe3c14734d83ef` | `libero_10 / LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 40 | put both the alphabet soup and the butter in the basket |
| `pro_0b1c687e3ddc97a406c8` | `libero_goal / open_the_middle_drawer_of_the_cabinet` | 1 | open the bottom drawer of the cabinet |
| `pro_199a9f469959b37b69c8` | `libero_goal / open_the_middle_drawer_of_the_cabinet` | 40 | open the top drawer of the cabinet |
| `pro_548975e5c701d1aa6f15` | `libero_goal / open_the_top_drawer_and_put_the_bowl_inside` | 40 | Open the top layer of the drawer and put the cream cheese inside |
| `pro_63de6689fb0d8bb9a7d3` | `libero_goal / open_the_top_drawer_and_put_the_bowl_inside` | 16 | Open the middle layer of the drawer and put the bowl inside |
| `pro_b7afc52d80a4aca85a1c` | `libero_goal / push_the_plate_to_the_front_of_the_stove` | 40 | Push the cream cheese to the front of the stove |
| `pro_37b3c7bc4038a8c0762d` | `libero_goal / put_the_bowl_on_the_plate` | 40 | Put the wine bottle on the plate |
| `pro_e740732c7feef94db754` | `libero_goal / put_the_bowl_on_the_stove` | 1 | Put the plate on the stove |
| `pro_0c6b763c2973cf01be99` | `libero_goal / put_the_bowl_on_top_of_the_cabinet` | 6 | Put the plate on the top of the drawer |
| `pro_9340097ec8a68dcf504e` | `libero_goal / put_the_cream_cheese_in_the_bowl` | 40 | put the wine bottle in the bowl |
| `pro_cfa2a5caa7f166d880b4` | `libero_goal / put_the_cream_cheese_in_the_bowl` | 40 | put the cream cheese on the top of the drawer |
| `pro_0fba16a9bf5ba7a7e03b` | `libero_goal / put_the_wine_bottle_on_the_rack` | 40 | Put the cream cheese on the rack |
| `pro_c8fb8f6775df1ce1d4b5` | `libero_goal / put_the_wine_bottle_on_the_rack` | 40 | Put the wine bottle on the stove |
| `pro_a536298f9c6861a8ffdc` | `libero_goal / turn_on_the_stove` | 40 | Turn off the stove |
| `pro_6edc47e423ffd9ac6574` | `libero_object / pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 40 | Pick the cream cheese and place it in the basket |
| `pro_3b53435b801e13dd927b` | `libero_object / pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 40 | Pick the ketchup and place it in the basket |
| `pro_774628767026067e51b8` | `libero_object / pick_up_the_butter_and_place_it_in_the_basket` | 40 | Pick the orange juice and place it in the basket |
| `pro_ffda6a06b2e133d58655` | `libero_object / pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 38 | Pick the salad dressing and place it in the basket |
| `pro_08f71634864df1e64204` | `libero_object / pick_up_the_cream_cheese_and_place_it_in_the_basket` | 40 | Pick the alphabet soup and place it in the basket |
| `pro_f671c284c66b54d9e36c` | `libero_object / pick_up_the_ketchup_and_place_it_in_the_basket` | 40 | Pick the milk and place it in the basket |
| `pro_a792c36ed0af9c458dff` | `libero_object / pick_up_the_milk_and_place_it_in_the_basket` | 40 | Pick the butter and place it in the basket |
| `pro_b09e867776d2e356f9b3` | `libero_object / pick_up_the_orange_juice_and_place_it_in_the_basket` | 40 | Pick the chocolate pudding and place it in the basket |
| `pro_eb9c191a9826a8e30a88` | `libero_object / pick_up_the_salad_dressing_and_place_it_in_the_basket` | 40 | Pick the tomato sauce and place it in the basket |
| `pro_97569f46806176984cc5` | `libero_object / pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 40 | Pick the bbq sauce and place it in the basket |
| `pro_11de0d41222d75530bff` | `libero_spatial / pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate` | 40 | Pick the akita black bowl not between the plate and the ramekin and place it on the plate |
| `pro_5e6c393c07fd6e0e0f95` | `libero_spatial / pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate` | 40 | Pick the akita black bowl next to the plate and place it on the plate |
| `pro_9be624ee28fcfb6c2c9b` | `libero_spatial / pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate` | 34 | Pick the akita black bowl on the top of the wooden cabinet and place it on the plate |
| `pro_4204b77f99e3403fa328` | `libero_spatial / pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate` | 38 | Pick the akita black bowl on the stove and place it on the plate |
| `pro_c6fbc12524b3671049c1` | `libero_spatial / pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate` | 40 | Pick the akita black bowl next to the ramekin and place it on the plate |
| `pro_a6fb3e45f8fc64a7026f` | `libero_spatial / pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate` | 40 | Pick the akita black bowl next to the cookie box and place it on the plate |
| `pro_aed8fa90746a6649c93d` | `libero_spatial / pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate` | 40 | Pick the akita black bowl on the cookie box and place it on the plate |
| `pro_7fac51859ae50f5cf8e8` | `libero_spatial / pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate` | 40 | Pick the akita black bowl on the stove and place it on the plate |

</details>

<details>
<summary>Подстановки объектов — 44 задач, 1698 эпизодов</summary>

| task_program_id | Источник BDDL: suite / stem | Эпизоды | Инструкция |
|---|---|---:|---|
| `pro_5460765259eb7fb55f91` | `libero_10 / LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket` | 39 | put both the butter and the cream cheese box in the basket |
| `pro_c4e39a40e297ba00a398` | `libero_10 / LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket` | 39 | put both the chocolate pudding and the tomato sauce in the basket |
| `pro_2e97135181decc4679f9` | `libero_10 / LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket` | 35 | put both the chocolate pudding and the butter in the basket |
| `pro_8778b6af79e9e39a5718` | `libero_10 / LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate` | 11 | put the white mug on the plate and put the alphabet soup to the right of the plate |
| `pro_f28b43e54b9d4fe2584c` | `libero_90 / KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet` | 39 | put the alphabet soup in the top drawer of the cabinet |
| `pro_1bb1ef54b8cf3f609879` | `libero_90 / LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket` | 38 | pick up the butter and put it in the basket |
| `pro_ef1fd0dca26924e30852` | `libero_90 / LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket` | 38 | pick up the butter and put it in the basket |
| `pro_83a5f8b10e8c7a7d52ed` | `libero_90 / LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket` | 40 | pick up the butter and put it in the basket |
| `pro_c21815bfc08ccf82f428` | `libero_90 / LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket` | 37 | pick up the butter and put it in the basket |
| `pro_dee0d964abf2123b07fd` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket` | 38 | pick up the chocolate pudding and put it in the basket |
| `pro_7ac3d77e3a427e520d60` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket` | 37 | pick up the chocolate pudding and put it in the basket |
| `pro_a543299ed0d6a31a6a49` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket` | 37 | pick up the chocolate pudding and put it in the basket |
| `pro_daa98bd3c635f0cff8b5` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket` | 39 | pick up the chocolate pudding and put it in the basket |
| `pro_f706f396c20c09b50534` | `libero_90 / LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket` | 37 | pick up the chocolate pudding and put it in the basket |
| `pro_369c7231cfe8dc91adf7` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray` | 40 | pick up the chocolate pudding and put it in the tray |
| `pro_effe85afcc229e7c10c9` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray` | 40 | pick up the chocolate pudding and put it in the tray |
| `pro_1f80840aa6b3b1867efe` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray` | 40 | pick up the chocolate pudding and put it in the tray |
| `pro_2cd965776ddf8593a7fd` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray` | 39 | pick up the chocolate pudding and put it in the tray |
| `pro_d4b18070d7e75d42c447` | `libero_90 / LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray` | 40 | pick up the chocolate pudding and put it in the tray |
| `pro_b2ad4649b70cd5177136` | `libero_90 / LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray` | 40 | pick up the alphabet soup and put it in the tray |
| `pro_e41423fd21aa6d43151f` | `libero_90 / LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate` | 39 | put the alphabet soup to the left of the plate |
| `pro_e51ebe03c4da8d1e99a0` | `libero_90 / LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate` | 39 | put the alphabet soup to the right of the plate |
| `pro_262681c7647982ebb5d1` | `libero_goal / put_the_cream_cheese_in_the_bowl` | 40 | Put the alphabet soup on the bowl |
| `pro_90f8875eb702c2ae805a` | `libero_goal / put_the_cream_cheese_in_the_bowl` | 40 | Put the butter on the bowl |
| `pro_c7c4644b8e00b858efef` | `libero_object / pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 39 | Pick the chocolate pudding and place it in the basket |
| `pro_ea55eecbc06164ca8f81` | `libero_object / pick_up_the_alphabet_soup_and_place_it_in_the_basket` | 40 | Pick the ketchup and place it in the basket |
| `pro_2ad6d385115ad8be2708` | `libero_object / pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 40 | Pick the butter and place it in the basket |
| `pro_c5464ded8fade32e9c2a` | `libero_object / pick_up_the_bbq_sauce_and_place_it_in_the_basket` | 39 | Pick the tomato sauce and place it in the basket |
| `pro_08b0e3a77de862652fa5` | `libero_object / pick_up_the_butter_and_place_it_in_the_basket` | 40 | Pick the alphabet soup and place it in the basket |
| `pro_ee3722f7f21decc70b00` | `libero_object / pick_up_the_butter_and_place_it_in_the_basket` | 40 | Pick the cream cheese and place it in the basket |
| `pro_1dd24d3ebf4487f45afd` | `libero_object / pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 40 | Pick the butter and place it in the basket |
| `pro_d08fa57efc2131b74152` | `libero_object / pick_up_the_chocolate_pudding_and_place_it_in_the_basket` | 40 | Pick the cream cheese and place it in the basket |
| `pro_34579e0714b1494af533` | `libero_object / pick_up_the_cream_cheese_and_place_it_in_the_basket` | 40 | Pick the chocolate pudding and place it in the basket |
| `pro_fcc32d75606c58517765` | `libero_object / pick_up_the_cream_cheese_and_place_it_in_the_basket` | 40 | Pick the ketchup and place it in the basket |
| `pro_522a8e8d5c71605d7977` | `libero_object / pick_up_the_ketchup_and_place_it_in_the_basket` | 40 | Pick the butter and place it in the basket |
| `pro_9e38b3b557f0bde91681` | `libero_object / pick_up_the_ketchup_and_place_it_in_the_basket` | 40 | Pick the chocolate pudding and place it in the basket |
| `pro_1b6f2f2279bcb0702737` | `libero_object / pick_up_the_milk_and_place_it_in_the_basket` | 40 | Pick the ketchup and place it in the basket |
| `pro_f6ecccecd917c0a0a0e1` | `libero_object / pick_up_the_milk_and_place_it_in_the_basket` | 40 | Pick the alphabet soup and place it in the basket |
| `pro_0c554f9f3901c91eeced` | `libero_object / pick_up_the_orange_juice_and_place_it_in_the_basket` | 39 | Pick the cream cheese and place it in the basket |
| `pro_bbc31fd3dfb462434197` | `libero_object / pick_up_the_orange_juice_and_place_it_in_the_basket` | 40 | Pick the alphabet soup and place it in the basket |
| `pro_2b79210b4c08f264907d` | `libero_object / pick_up_the_salad_dressing_and_place_it_in_the_basket` | 40 | Pick the chocolate pudding and place it in the basket |
| `pro_79e687be514d260e252f` | `libero_object / pick_up_the_salad_dressing_and_place_it_in_the_basket` | 40 | Pick the butter and place it in the basket |
| `pro_0aec33f731b6a3b27dc6` | `libero_object / pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 40 | Pick the cream cheese and place it in the basket |
| `pro_c10da3bded33b1f08541` | `libero_object / pick_up_the_tomato_sauce_and_place_it_in_the_basket` | 40 | Pick the alphabet soup and place it in the basket |

</details>

<details>
<summary>Составные программы — 20 задач, 604 эпизодов</summary>

| task_program_id | Источник BDDL: suite / stem | Эпизоды | Инструкция |
|---|---|---:|---|
| `pro_199e51f61d36c0e96362` | `libero_90 / KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` | 32 | Put the black bowl on the top of the wooden cabinet, then put the black bowl inside the top drawer of the cabinet |
| `pro_04b4f1152e1aea1d0f3c` | `libero_90 / KITCHEN_SCENE1_put_the_black_bowl_on_the_plate` | 36 | Put the black bowl on the top of the wooden cabinet, then put the black bowl on the plate |
| `pro_57d7da83c3e01e21c2b2` | `libero_90 / KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet` | 40 | Put the black bowl on the plate, then put the black bowl on the top of the wooden cabinet |
| `pro_969e1619acc7f202484a` | `libero_90 / KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet` | 40 | Put the black bowl on the top of the white cabinet, then put the black bowl inside the bottom drawer of the cabinet |
| `pro_023bb3e3c159121382ef` | `libero_90 / KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet` | 16 | Put the wine bottle on the top of the white cabinet, then put the wine bottle inside the bottom drawer of the cabinet |
| `pro_61516d40c9c3477157ac` | `libero_90 / KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack` | 23 | Put the wine bottle on the top of the white cabinet, then put the wine bottle on the wine rack top region |
| `pro_ebd1a92650c75e573ca7` | `libero_90 / KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet` | 40 | Put the black bowl on the plate, then put the black bowl inside the top drawer of the cabinet |
| `pro_cfe2a37427f512d32acc` | `libero_90 / KITCHEN_SCENE5_put_the_black_bowl_on_the_plate` | 35 | Put the black bowl on the top of the white cabinet, then put the black bowl on the plate |
| `pro_07228af4e49f34457b0e` | `libero_90 / KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet` | 39 | Put the black bowl on the plate, then put the black bowl on the top of the white cabinet |
| `pro_29330a1bb6b9df846719` | `libero_90 / KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet` | 5 | Put the white bowl on the stove, then put the white bowl on the top of the wooden two layer shelf |
| `pro_30e28c5448dea8cba910` | `libero_goal / open_the_top_drawer_and_put_the_bowl_inside` | 1 | Put the black bowl on the plate, then put the black bowl inside the top drawer of the cabinet |
| `pro_52feaf0a14f43af85f56` | `libero_goal / put_the_bowl_on_the_plate` | 32 | Put the black bowl on the plate, then put the black bowl on the top of the wooden cabinet, then put the black bowl on the plate |
| `pro_62cdd0c2a2617e29e96f` | `libero_goal / put_the_bowl_on_the_plate` | 37 | Put the black bowl on the top of the wooden cabinet, then put the black bowl on the plate |
| `pro_2b5689af683396afd98c` | `libero_goal / put_the_bowl_on_the_stove` | 38 | Put the black bowl on the plate, then put the black bowl on the stove |
| `pro_ff74bef25bd26be5ccbb` | `libero_goal / put_the_bowl_on_the_stove` | 39 | Put the black bowl on the stove, then put the black bowl on the plate, then put the black bowl on the stove |
| `pro_0b035448de1d8120d151` | `libero_goal / put_the_bowl_on_top_of_the_cabinet` | 40 | Put the black bowl on the plate, then put the black bowl on the top of the wooden cabinet |
| `pro_7bd606400a3b8ca9c40a` | `libero_goal / put_the_bowl_on_top_of_the_cabinet` | 27 | Put the black bowl on the top of the wooden cabinet, then put the black bowl on the plate, then put the black bowl on the top of the wooden cabinet |
| `pro_22c7581db20518f0ddf7` | `libero_goal / put_the_wine_bottle_on_the_rack` | 40 | Put the wine bottle on the plate, then put the wine bottle on the wine rack top region |
| `pro_4be50292c72a6204ba02` | `libero_goal / put_the_wine_bottle_on_top_of_the_cabinet` | 39 | Put the wine bottle on the plate, then put the wine bottle on the top of the wooden cabinet |
| `pro_e613b4ac5cae76e26705` | `libero_goal / put_the_wine_bottle_on_top_of_the_cabinet` | 5 | Put the wine bottle on the top of the wooden cabinet, then put the wine bottle on the plate, then put the wine bottle on the top of the wooden cabinet |

</details>
