# Hypernetwork-LoRA для SmolVLA

Гиперсеть (HN), которая **за один forward генерирует LoRA-адаптер** для замороженной
VLA-политики SmolVLA по условию задачи — тексту инструкции, кадру наблюдения или
демонстрациям. На тесте адаптация не требует градиентных шагов: условие → гиперсеть →
веса адаптера → политика действует.

Идея восходит к Hyper-LoRA (Sharma, Xiong et al., NeurIPS 2024 FT Workshop):
бэкбон заморожен, HN генерирует LoRA на MLP-слои, голова действий доучивается.
Здесь рецепт перенесён с Octo на SmolVLA и расширен кондиционированием на кадры и
демонстрации + оценкой на OOD-осях LIBERO-Pro.

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
  setup.sh                   бутстрап окружения с нуля
  train.sh / eval.sh         единые точки запуска; все ручки — env-переменными,
                             документированы в шапках самих скриптов
  render_recolor_clips.py    sim-аугментации демо: MuJoCo state-replay с перекраской
                             объектов / сдвигом камеры (движение то же, вид другой)
  build_xpair_cache.py       оффлайн-кеш эмбеддингов демонстраций (encode-once)
  build_frame_bank.py        банк t=0 кадров всех эпизодов
  patch_lerobot.py           пост-инсталл фикс lerobot под transformers 5 / py3.12
  analyze_lora.py            зонд: зависят ли сгенерированные LoRA от задачи
```

Конвенция кода: **`src/traj_data/` не импортирует lerobot** — чистые numpy/torch
модули с узкими интерфейсами; весь lerobot-специфичный код живёт в `modeling_*` и
скриптах (в скриптах «чистая» логика — верхнеуровневые функции, GPU/датасеты — в
`main()`).

## Установка

```bash
git clone <repo> && cd fewshot_vla
bash scripts/setup.sh
```

Версии в `requirements.txt` запинены не случайно — каждый пин закреплён после
реального падения (`lerobot==0.5.1`, `transformers==5.7.0`, `mujoco<3.3`); не
обновляйте без проверки. `HF_HUB_OFFLINE=1` несовместим с lerobot 0.5.1.
Headless-рендер — EGL (выставляется в скриптах).

## Подготовка данных (нужна только для traj-режима и vision-абляций)

Датасет `lerobot/libero` скачивается автоматически. Для демо-кондиционирования:

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
по умолчанию — **в шапках `scripts/train.sh` и `scripts/eval.sh`** (они и есть
источник правды; конкретные сетапы экспериментов меняются и здесь не фиксируются).

```bash
MODE=vision bash scripts/train.sh                       # обучение (см. шапку train.sh)
MODE=traj ENC=dino OUTPUT=outputs/my_run bash scripts/train.sh

EPISODE_CACHE=1 POLICIES="my_run=outputs/my_run/checkpoints/last/pretrained_model" \
    bash scripts/eval.sh                                # eval-матрица, резюмируемая
```

Результаты: `outputs/eval_matrix/<label>/<suite>/seed_<s>/eval_info.json`
(`overall.pc_success`), сводная таблица печатается в конце eval.sh.
Логи обучения: `TB=1` → TensorBoard в `$OUTPUT/tensorboard` (без сети), либо
`WANDB_OFFLINE=1`.

## Частые вопросы

- **Почему два класса политик?** `traj_hyper_lora_smolvla` наследует
  `hyper_lora_smolvla`; с выключенными флагами структурно идентичен ему, так что
  traj-стек не трогает text/vision-код и чекпоинты.
- **Почему кеш ragged?** Демо разной длины; конкат токенов одним mmap +
  (offset, length) на запись — ленивое чтение без паддинга на диске.
- **Откуда HN знает текст?** Токенизация и эмбеддинг — от самого VLM (общие с
  политикой), HN учит только проекцию в своё скрытое пространство.
- **Диагностика:** `[TRAJ] task=...` в eval-логе — резолв задачи и подгрузка демо;
  `HN_LOG_LORA=1` — норма/дрейф генерируемых весов; при «зависании» после
  «Creating policy» смотрите `nvidia-smi` и `py-spy dump` (типично: wandb без сети
  или гонка HF-кеша при параллельных стартах).
