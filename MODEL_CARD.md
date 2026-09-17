# LIBERO-90 для lerobot + SmolVLA, обученная на его train-части

Два артефакта на Hugging Face:

| | |
|---|---|
| датасет | [`Kesvill/libero_90_lerobot_v3`](https://huggingface.co/datasets/Kesvill/libero_90_lerobot_v3) — LIBERO-90 в формате lerobot v3.0, готовый к `lerobot/libero`-совместимому трейну и эвалу |
| модель | [`Kesvill/smolvla_libero_90`](https://huggingface.co/Kesvill/smolvla_libero_90) — SmolVLA (`lerobot/smolvla_base`), дообученная по рецепту статьи на 39 задачах train-сплита |

Код: репо `fewshot_vla` (`bash scripts/setup.sh && source venv/bin/activate`, команды из корня).

## Датасет

**Что внутри.** Оригинальные демонстрации LIBERO-90 (90 задач, 20 сцен) после no-op фильтра
OpenVLA — 3921 эпизод, 569 249 кадров (задача 51 `pick up the butter and put it in the basket`
фильтром выпала целиком). Видео AV1 256×256, обе камеры. Схема как у `lerobot/libero`:

| ключ | форма | смысл |
|---|---|---|
| `observation.images.image` | 256×256×3 | agentview |
| `observation.images.image2` | 256×256×3 | wrist |
| `observation.state` | 8 | eef xyz + axis-angle + 2 гриппер |
| `action` | 7 | OSC delta xyz/rpy + гриппер **+1 = закрыть, −1 = открыть** (конвенция robosuite/lerobot/env) |

Схема, `fps` (10), `robot_type` (panda) и `names` фич — **побайтово как у `lerobot/libero`**, поэтому оба
датасета можно объединять штатным `lerobot_edit_dataset --operation.type merge` и учить/эвалить вместе
(в lerobot 0.5.1 список `repo_id` в трейнере не поддерживается — совместный трейн = один смерженный датасет).
Кадры — в конвенции env-обёртки lerobot (то, что политика видит на эвале): никаких
поворотов и флипов при обучении и эвале не нужно.

**Сплит.** В корне датасета:

- `train_episodes.json` — 1733 эпизода, 39 задач (train-40 из `configs/libero90_split.json` минус 51);
- `eval_episodes.json` — 2188 эпизодов, 50 held-out задач;
- `episode_task_map.json` — `episode_index → task_id` бенчмарка (порядок `libero_task_map["libero_90"]`, он же `--env.task_ids` у lerobot-eval).

Сплит сделан сидом 42 со стратификацией по 20 сценам; 14 задач LIBERO-90, которые являются
подшагами задач LIBERO-10, принудительно в train. **Не ключуйте задачи по тексту**: у 90 задач
только 74 уникальные инструкции, `task_index` датасета их склеивает — используйте `task_id`.

**Как собран** (провенанс, повторять не нужно): `scripts/prepare_nvidia_libero90.py` —
`nvidia/LIBERO_LeRobot_v3/libero_90` (ревизия `e590737`) → `wrist_image→image2` → гриппер
`0/1 → ±1` в данных и stats → `task_id` по побайтовому совпадению последовательностей действий с
`yzembodied/libero_90_image` (у NVIDIA нумерация эпизодов не по задачам) → проверка ориентации
кадров против эталона входа политики; затем `scripts/harmonize_libero90.py` — приведение к конвенциям
lerobot/libero (метка fps 20→10 с ремуксом видео без перекодирования, лишние `observation.states.*` убраны).

## Как с ним работать (сценарии)

Один раз:
```bash
hf download Kesvill/libero_90_lerobot_v3 --repo-type dataset --local-dir outputs/libero90/libero_90_lerobot_v3
```

**1. Эвал любой политики на LIBERO-90 + дефолтных сьютах.** `libero_90_train` / `libero_90_eval` —
псевдосьюты `scripts/eval.sh` (стоковый lerobot-eval `--env.task=libero_90` с `task_ids` из
`configs/libero90_split.json`, чанками по 10 задач — env всех выбранных задач живут одновременно):
```bash
POLICIES="my=<путь или HF id>" TASKS="libero_90_train libero_90_eval libero_10 libero_goal libero_object libero_spatial" \
SEEDS=1000 BATCH=10 OUT_ROOT=outputs/my_eval bash scripts/eval.sh
python scripts/summarize_matrix.py outputs/my_eval --per_task libero_90_eval
```
`BATCH` — параллельных env на задачу (80 ГБ → 10, 24 ГБ → 2). Матрица резюмируемая. На несколько
карт — `scripts/eval_all_8gpu.sh` (в шапке пример раскладки `ASSIGN`).

**2. Файнтьюн SmolVLA на train-части по нашему рецепту** (так получена `smolvla_libero_90`):
```bash
python scripts/prepare_base_smolvla_libero.py --out outputs/base/smolvla_base_libero_io
NPROC=8 STEPS=100000 OUT=outputs/my_run bash scripts/finetune_ours.sh     # NPROC карт, глобальный batch 64
```
Ручки через env: `BASE_PATH` (старт с другой базы, в т.ч. `Kesvill/smolvla_libero_90`), `DATA_ROOT`,
`EPISODES_FILE` (по умолчанию `train_episodes.json` датасета), `BATCH`, `GPUS`, `WORKERS`; смоук —
`STEPS=300 OUT=outputs/probe`. Лог — `<OUT>.log`, TensorBoard — `<OUT>/tensorboard`.

**3. Стоковый lerobot-train любой политикой на train-части:**
```bash
EPIS="$(python -c 'import json;print(json.dumps(json.load(open("outputs/libero90/libero_90_lerobot_v3/train_episodes.json")),separators=(",",":")))')"
python train_hyper_lora.py --policy.type=<...> --dataset.repo_id=Kesvill/libero_90_lerobot_v3 \
    --dataset.root=outputs/libero90/libero_90_lerobot_v3 --dataset.episodes="$EPIS" --dataset.video_backend=pyav ...
```
`train_hyper_lora.py` — обёртка над `lerobot-train` с теми же флагами; в ней починены нестабильный
fingerprint кеша datasets и построчный декод картинок при инициализации (голый lerobot-train на этом
датасете стартует 20 мин на процесс и виснет в DDP). `video_backend=pyav` — AV1 не всякий torchcodec
декодирует. Без `--dataset.episodes` в обучение попадут и 50 held-out задач.

**4. Совместный трейн с дефолтным LIBERO** — один смерженный датасет (в lerobot 0.5.1 список
`repo_id` в трейнере не поддерживается):
```bash
python scripts/check_libero_merge.py --merge     # validate_all_metadata + merge -> outputs/libero90/libero_all
```
В смерженном датасете наши эпизоды идут после 1693 эпизодов lerobot/libero (train-список сдвигается
на 1693), а `task_index` строится по тексту инструкции — эпизоды выбирайте по `episode_task_map.json`,
не по тексту.

**5. Гиперсеть на LIBERO-90** — пока не поддержана: резолв задачи по тексту и кеши демо в
`src/traj_data` рассчитаны на lerobot/libero (74 уникальных текста на 90 задач ломают ключевание);
это следующий шаг этапа.

## Модель `smolvla_libero_90`

| | |
|---|---|
| база | `lerobot/smolvla_base`: SmolVLM2-500M-Video-Instruct, 16 слоёв LLM, action expert ×0.75 (hidden 720, 16 слоёв) |
| данные | train-часть датасета выше: 39 задач, 1733 эпизода |
| рецепт | статья SmolVLA §4.3: VLM заморожен, учится только action expert (+ state/action-проекции); 100k шагов, глобальный batch 64 (8×A100, по 8 на карту), lr 1e-4 → cosine 2.5e-6 на весь горизонт, bf16, картинки 512×512, chunk 50 |
| I/O | `image`, `image2`, `state[8]` → `action[7]`; `n_action_steps=1` запечён в `config.json` |
| воспроизвести | `python scripts/prepare_base_smolvla_libero.py --out outputs/base/smolvla_base_libero_io` (копия smolvla_base с LIBERO-фичами), затем `NPROC=8 STEPS=100000 OUT=outputs/my_run bash scripts/finetune_ours.sh` |

Загрузка: `--policy.path=Kesvill/smolvla_libero_90` или `SmolVLAPolicy.from_pretrained(...)`.

**Результаты** (success rate %, 50 эпизодов/задачу, seed 1000, 10 flow-шагов):

| сьют | задач | SR |
|---|---|---|
| libero_90_train (виденные) | 40 | **74.8** |
| libero_90_eval (held-out, те же сцены) | 50 | **1.9** |
| libero_10 | 10 | 0.2 |
| libero_goal | 10 | 0.0 |
| libero_object | 10 | 0.0 |
| libero_spatial | 10 | 0.0 |
| LIBERO-Pro libero_10 lan / object / swap / task | 10 | 0.0 / 0.0 / 0.0 / 7.6 |

Задача 51 в `libero_90_train` не входила в обучение (нет в датасете) и даёт 64% — перенос
внутри сцены от задач 50/54. Для сравнения `HuggingFaceVLA/smolvla_libero` (училась на 4
сьютах): libero_10 41.8, goal 84.2, object 91.0, spatial 77.4; Pro lan 0.0, object 18.8,
swap 0.0, task 0.0. Вывод: база, обученная на 40 задачах LIBERO-90, не переносится ни на новые
задачи в виденных сценах, ни на другие сьюты — это пространство для методов адаптации.
