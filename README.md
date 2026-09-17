# Hypernetwork-LoRA для SmolVLA

Гиперсеть (HN), которая **за один forward генерирует LoRA-адаптер** для замороженной
VLA-политики SmolVLA по условию задачи — тексту инструкции, кадру наблюдения или
демонстрациям. На тесте адаптация не требует градиентных шагов: условие → гиперсеть →
веса адаптера → политика действует.


## Как это работает

```
условие ──────────────► HyperNetwork ─────────► LoRA ΔW ──────► SmolVLA
(текст / кадр / демо)   concat self-attention   на MLP-слои     VLM заморожен;
                        [32 layer-query |       замороженного   action expert обучается
                         текст | условие]       VLM             (flow matching → действия)
```

- Вход HN — одна последовательность токенов: 32 обучаемых *layer-query* (по одному
  на слой VLM) + токены инструкции (эмбеддинги самого VLM + обучаемая проекция) +
  токены условия (зависят от режима).
- После self-attention первые 32 позиции декодируются в LoRA-матрицы двумя
  **общими для всех слоёв** линейными головами (`W_up` zero-init ⇒ старт из базовой
  политики).
- `dynamic_lora.py` подставляет сгенерированные веса в Linear-слои VLM на время
  forward — они не являются параметрами модуля.
- Обучение — обычный flow-matching лосс политики на `lerobot/libero`; учатся
  гиперсеть и action expert, VLM всегда заморожен.
- На eval адаптер обычно строится один раз на старте эпизода и замораживается
  (`EPISODE_CACHE=1`).

Режимы кондиционирования (`MODE` в train.sh): `text` — только инструкция; `vision` —
инструкция + кадр (SigLIP-токены VLM / патчи DINOv2); `traj` — инструкция + целые
траектории демонстраций (покадровые эмбеддинги DINO или V-JEPA2 напрямую в
attention); `lora` — базлайн с обычным PEFT-LoRA.

## Карта репозитория

```
train_hyper_lora.py        тренинг: обёртка над lerobot-train (регистрация политик,
                           сверка конфига с базовым чекпоинтом, TensorBoard-логгер)
eval_hyper_lora.py         eval: регистрация политик + LIBERO-Pro сьютов → lerobot-eval
eval_libero_pro.py         то же для чекпоинтов без гиперсети

src/hyper_lora/            политика №1: text/vision-кондиционирование
  hypernetwork.py            ядро HN: стримы → self-attention → головы весов
  dynamic_lora.py            обёртка Linear с подставными LoRA-весами
  modeling_...py             lerobot-glue: патчинг VLM, инжект адаптера, режимы кадра
  configuration_...py        конфиг (hn_* ручки); base_config.py — сверка с базой

src/hyper_lora_traj/       политика №2 (наследует №1): демо-кондиционирование
  fusion_hypernetwork.py     HN + стрим траекторий (сырые токены + маркеры границ демо)
  modeling_...py             train: селектор контекста; eval: резолв задачи → демо из кеша
  configuration_...py        ручки traj-режима

src/traj_data/             чистые модули БЕЗ lerobot (данные условия)
  encoder.py                 энкодеры клипов: DINOv2 (CLS/кадр), V-JEPA2 (mean-pool/tubelet)
  cache_io.py / traj_cache.py  ragged-кеш эмбеддингов демо (tokens.mmap + index.json)
  xpair_select.py            выбор контекста на train/eval, паддинг и маски
  frame_bank.py              банк стартовых кадров для vision-абляций
  augment.py / diagnostics.py  вспомогательные

src/libero_pro/            LIBERO-Pro поверх родного LIBERO (сьюты, перекрашенные объекты)
src/data/libero.py         обход бага file_index в lerobot/libero

scripts/
  setup.sh                   бутстрап окружения с нуля (PYTHON_VERSION=3.12)
  train.sh / eval.sh         единые точки запуска HN/LoRA/эвала; все ручки — env-переменными,
                             документированы в шапках самих скриптов
  eval_all_8gpu.sh           та же eval-матрица, разложенная по картам (ASSIGN — ряд на карту)
  summarize_matrix.py        сводка eval-матрицы (объединяет чанки libero_90_*, per-task таблица)
  finetune_ours.sh           файнтьюн базы SmolVLA по рецепту статьи одной командой
  args_finetune_ours.sh      его argv (датасет, train-эпизоды, база); source-ится лаунчером
  prepare_base_smolvla_libero.py  копия lerobot/smolvla_base с LIBERO-фичами (старт файнтьюна)
  push_hf.py                 залить чекпоинт / датасет на Hub (датасету ставит тег v3.0)
  check_libero_merge.py      проверка совместимости с lerobot/libero + merge в один датасет
  render_recolor_clips.py    sim-аугментации демо: MuJoCo state-replay с перекраской
                             объектов / сдвигом камеры (движение то же, вид другой)
  build_xpair_cache.py       оффлайн-кеш эмбеддингов демонстраций (encode-once)
  build_frame_bank.py        банк t=0 кадров всех эпизодов
  make_libero90_split.py     фикс. сплит LIBERO-90: 40 train / 50 eval (configs/libero90_*.json)
  libero90_episodes.py       чанки task_ids псевдосьютов libero_90_train/eval для eval.sh
  prepare_nvidia_libero90.py / harmonize_libero90.py / flip_libero90_images.py /
  rename_libero90_features.py  провенанс датасета Kesvill/libero_90_lerobot_v3 (см. MODEL_CARD)
  patch_lerobot.py           пост-инсталл фикс lerobot под transformers 5 / py3.12
  analyze_lora.py            зонд: зависят ли сгенерированные LoRA от задачи
```

 **`src/traj_data/` не импортирует lerobot** — чистые numpy/torch
модули; весь lerobot-специфичный код живёт в `modeling_*` и
скриптах.

## Установка

```bash
git clone <repo> && cd fewshot_vla
export PYTHON_VERSION=3.12   # lerobot 0.5.1 требует 3.12; в некоторых образах переменная уже = 3.11
bash scripts/setup.sh
source venv/bin/activate
```


## Подготовка данных (нужна только для traj-режима и vision-абляций)

Датасет `lerobot/libero` скачивается автоматически. Сырые данные надо вручную скачивать только для генерации аугментаций:

```bash
# 1. HDF5-демо LIBERO с сырыми MuJoCo-состояниями (официальные ссылки мертвы):
hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
    --local-dir third_party/LIBERO/libero/datasets --include "libero_*/*"

# 2. sim-аугментации (replay состояний в изменённой сцене; резюмируемо, шардируется):
python scripts/render_recolor_clips.py --colors red,green --cam_jitters 2 --all_frames \
    --out outputs/rendered_recolor

# 3. кеш эмбеддингов демо + банк стартовых кадров:
python scripts/build_xpair_cache.py --encoder dino --rendered_dir outputs/rendered_recolor --out outputs/xpair_cache/dino
python scripts/build_frame_bank.py  --rendered_dir outputs/rendered_recolor --out outputs/frame_bank.npz
```

## Запуск

Обучение и оценка конфигурируются env-переменными; их актуальный список и значения
по умолчанию — **в шапках `scripts/train.sh` и `scripts/eval.sh`**

```bash
MODE=vision bash scripts/train.sh                       # обучение (см. шапку train.sh)
MODE=traj ENC=dino OUTPUT=outputs/my_run bash scripts/train.sh

EPISODE_CACHE=1 POLICIES="my_run=outputs/my_run/checkpoints/last/pretrained_model" \
    bash scripts/eval.sh                                # eval-матрица, резюмируемая
```

Результаты: `outputs/eval_matrix/<label>/<suite>/seed_<s>/eval_info.json`
(`overall.pc_success`), сводная таблица печатается в конце eval.sh. Рядом же сохраняются видео с симулятора
Логи обучения: `TB=1` → TensorBoard в `$OUTPUT/tensorboard` (без сети), либо
`WANDB_OFFLINE=1`.

**Возможное падение на импорте из-за torchaudio.** Если запуск валится с
`OSError: libcudart.so.XX: cannot open shared object file` надо удалить torchaudio:

```bash
pip uninstall -y torchaudio
```

## Этап LIBERO-90 held-out

Датасет и базовая модель лежат на Hugging Face; всё, что нужно команде для работы с ними, —
в `MODEL_CARD.md` (что внутри, конвенции, сценарии трейна/эвала вместе с дефолтными сьютами):
[`Kesvill/libero_90_lerobot_v3`](https://huggingface.co/datasets/Kesvill/libero_90_lerobot_v3) ·
[`Kesvill/smolvla_libero_90`](https://huggingface.co/Kesvill/smolvla_libero_90).
Идея этапа (протокол: `docs/experiments/2026-08-30-plan-libero90-heldout.md` в основном репо):
база учится только на train-сплите LIBERO-90 (40 задач по `configs/libero90_split.json`; в датасете
после no-op фильтра их 39 — задачи 51 нет), остальные 50 задач + libero_10/goal/object/spatial —
held-out; дальше гиперсеть при полностью замороженной базе предсказывает LoRA только на action
expert (`LORA_TARGET=expert_mlp`).

```bash
# данные + база (один раз)
hf download Kesvill/libero_90_lerobot_v3 --repo-type dataset --local-dir outputs/libero90/libero_90_lerobot_v3
python scripts/prepare_base_smolvla_libero.py --out outputs/base/smolvla_base_libero_io

# файнтьюн базы на train-части: рецепт статьи (100k шагов, глобальный batch 64, lr 1e-4 cosine -> 2.5e-6, bf16)
NPROC=8 STEPS=100000 OUT=outputs/smolvla_libero_90 bash scripts/finetune_ours.sh

# eval: libero_90_train/libero_90_eval — псевдосьюты (чанки по 10 задач) рядом с дефолтными
POLICIES="ours=Kesvill/smolvla_libero_90" \
TASKS="libero_90_train libero_90_eval libero_10 libero_goal libero_object libero_spatial" \
SEEDS=1000 BATCH=10 bash scripts/eval.sh
python scripts/summarize_matrix.py outputs/eval_matrix --per_task libero_90_eval
```

Результат базы (SR %, 50 эп./задачу): 90-train 74.8 · 90-eval 1.9 · libero_10 0.2 ·
goal/object/spatial 0.0 · Pro lan/object/swap 0.0, task 7.6 — таблица и разбор в `MODEL_CARD.md`.

Гиперсеть на этой базе **пока не запускать**: резолв задачи по тексту инструкции и кеши демо
(`build_xpair_cache.py`, `build_frame_bank.py`) рассчитаны на lerobot/libero; для LIBERO-90 нужен
ключ задачи по `task_id` (74 текста на 90 задач) и пересборка кешей — это следующий шаг этапа.
Ручки под него в `train.sh` уже есть: `BASE=`, `DATASET=`, `DATASET_ROOT=`, `VIDEO_BACKEND=pyav`,
`EPISODES=` (список эпизодов из `train_episodes.json`), `LORA_TARGET=expert_mlp`.

Провенанс датасета (повторять не нужно): `scripts/prepare_nvidia_libero90.py` (NVIDIA no-op
LIBERO-90 → камеры → гриппер 0/1→±1 → task_id по совпадению действий с yzembodied → списки
эпизодов сплита → проверка ориентации кадров) и `scripts/harmonize_libero90.py` (fps/схема как у
lerobot/libero); `make_libero90_split.py`, `flip_libero90_images.py`, `rename_libero90_features.py`
— инструменты той же цепочки.

## FYI
Пока забейте и не смотрите на код связанный с ретривалом/кондишенингом траекторий целых, там буду переделывать

