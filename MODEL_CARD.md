# smolvla_libero_ours — SmolVLA, дообученная на LIBERO-90 (train-40)

Мини-инструкция для команды: что это за чекпоинт, где лежит, как его гонять и как
работать с нашим датасетом LIBERO-90, чтобы ставить свои эксперименты поверх.
Код: репо `fewshot_vla` (все команды — из его корня, `source venv/bin/activate`).

## 1. Что это

| | |
|---|---|
| База | `lerobot/smolvla_base` (SmolVLM2-500M-Video-Instruct, **16 слоёв LLM**, action expert ×0.75 = hidden 720, 16 слоёв) |
| Данные | LIBERO-90, **40 задач** (train-сплит, 2000 демо); остальные 50 задач + libero_10/goal/object/spatial никогда не видела |
| Рецепт | статья SmolVLA §4.3: VLM заморожен, учится только action expert (+ state/action-проекции); 100k шагов, глобальный batch 64, lr 1e-4 → cosine 2.5e-6 на весь горизонт, bf16, картинки 512×512, chunk 50 |
| Эвал-протокол | `n_action_steps=1` (запечён в config.json), 10 flow-шагов, 50 эпизодов/задачу, seed 1000 |
| I/O | `observation.images.image` (agentview), `observation.images.image2` (wrist), `observation.state[8]`, `action[7]` — как у `HuggingFaceVLA/smolvla_libero` и env-обёртки lerobot |

Версии: **v2** (`smolvla_libero_ours_v2`) обучена на исправленном датасете и работает
с env напрямую — используйте её. **v1** (`smolvla_libero_ours`) обучена на зеркальном
датасете (см. §4) и корректна только с `EVAL_FLIP_LR="observation.images.image observation.images.image2"`
на эвале — для новых экспериментов не брать.

## 2. Где лежит

```
<SHARE>/artifacts/smolvla_libero_ours_v2/pretrained_model/   # config.json, model.safetensors, policy_pre/postprocessor.json, train_config.json
<SHARE>/artifacts/libero_90_image_flipped/                   # датасет lerobot v3.0, 4500 эп., ~30 ГБ
<SHARE>/artifacts/libero90_split.json                        # фиксированный сплит 40/50 (копия configs/ из репо)
```
`<SHARE>` = `/home/jovyan/shares/SR004.nfs2/skripkin` (см. §6, как выложить/обновить).
Чекпоинт — обычный lerobot-чекпоинт: `SmolVLAPolicy.from_pretrained(path)` или `--policy.path=path`.

## 3. Результаты

Заполняется из `python scripts/summarize_matrix.py outputs/eval_matrix_v2 --per_task libero_90_eval`
(mean success rate, %, 50 эп./задачу, seed 1000):

| policy | 90-train (40 виденных) | 90-eval (50 held-out) | libero_10 | goal | object | spatial | Pro lan | Pro object | Pro swap | Pro task |
|---|---|---|---|---|---|---|---|---|---|---|
| smolvla_libero_ours_v2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Ориентиры: на 2 train-задачах × 20 эп. v1 с правильной конвенцией кадров дала 92.5–97.5%;
по 4 чанкам train — ~80%. Все OOD-сьюты у базы ожидаемо низкие — это и есть
пространство для методов адаптации (гиперсеть, LoRA по демо и т.п.).

## 4. Датасет LIBERO-90: конвенции и грабли

- Источник: `yzembodied/libero_90_image` (оригинальные 4500 демо LIBERO-90, v2.1) →
  `convert_dataset_v21_to_v30` → `scripts/rename_libero90_features.py` (wrist_image→image2)
  → **`scripts/flip_libero90_images.py` по обеим камерам**. Без флипа кадры зеркальны
  относительно рендера env: политика на них даёт 25–40% на своих же задачах вместо 80+.
  Не учитесь на `libero_90_image` (нефлипнутом) — только на `libero_90_image_flipped`.
- `episode_index // 50 == task_id` бенчмарка (порядок `libero_task_map["libero_90"]`,
  он же порядок `--env.task_ids` у lerobot-eval). Список train-эпизодов:
  `python scripts/libero90_episodes.py --part train` (2000 id для `--dataset.episodes`).
- **74 уникальные инструкции на 90 задач** (12 текстов делят 28 задач в разных сценах).
  Ключуйте задачи по `task_id`, не по тексту: `task_index` датасета и любой резолв «по
  инструкции» на LIBERO-90 подмешает чужую сцену.
- Метка `fps=20` в info.json — просто метка: кадров столько же, сколько в lerobot/libero
  (median 141 на эпизод, max 373 < горизонта 400). No-op кадров 0.5% — фильтр не нужен.
- Сплит `configs/libero90_split.json` (сид 42, стратификация по 20 сценам, 14 подзадач
  LIBERO-10 принудительно в train, поле `twin_ids` — текстовые близнецы). Псевдосьюты
  `libero_90_train` / `libero_90_eval` в `scripts/eval.sh` читают его сами.
- Первое открытие датасета строит arrow-кеш (~20 мин, десятки ГБ в `$HF_DATASETS_CACHE`);
  запускайте обучение через `train_hyper_lora.py` (обёртка над lerobot-train): в ней
  починены нестабильный fingerprint кеша, O(N)-цикл с декодом картинок при инициализации
  (20 мин на процесс, в DDP выглядел как вечный вис) и таймаут барьера. Голый
  `lerobot-train` на этом датасете эти грабли соберёт заново.

## 5. Как использовать

Эвал (одна карта, любой набор сьютов; матрица резюмируемая):
```bash
POLICIES="ours_v2=<SHARE>/artifacts/smolvla_libero_ours_v2/pretrained_model" \
TASKS="libero_90_eval libero_10 libero_goal libero_object libero_spatial" SEEDS=1000 BATCH=10 \
OUT_ROOT=outputs/my_eval bash scripts/eval.sh
```
Все 8 карт сразу: `bash scripts/eval_all_8gpu.sh ours_v2=<path>` (раскладка в `ASSIGN`).

Свой файнтьюн от нашей базы или от smolvla_base (рецепт статьи, одна короткая команда):
```bash
BASE_PATH=<SHARE>/artifacts/smolvla_libero_ours_v2/pretrained_model \
DATA_ROOT=<SHARE>/artifacts/libero_90_image_flipped \
NPROC=8 STEPS=100000 OUT=outputs/my_finetune bash scripts/finetune_ours.sh
```
(`NPROC` карт, глобальный батч 64 держится автоматически; `BATCH`, `GPUS`, `WORKERS` — ручки.)

Гиперсеть/LoRA поверх замороженной базы — `scripts/train.sh` с `BASE=<path>`,
`DATASET_ROOT=<...>/libero_90_image_flipped`, `LORA_TARGET=expert_mlp` (см. README репо).

## 6. Выложить/обновить артефакты на шару

```bash
SHARE=/home/jovyan/shares/SR004.nfs2/skripkin
mkdir -p $SHARE/artifacts/smolvla_libero_ours_v2
rsync -aL outputs/smolvla_libero_ours_v2/checkpoints/last/pretrained_model/ $SHARE/artifacts/smolvla_libero_ours_v2/pretrained_model/
rsync -a outputs/libero90/libero_90_image_flipped/ $SHARE/artifacts/libero_90_image_flipped/
cp configs/libero90_split.json $SHARE/artifacts/
cp docs/MODEL_CARD_smolvla_libero_ours.md $SHARE/artifacts/README.md   # в fewshot_vla файл лежит в корне как MODEL_CARD.md
chmod -R a+rX $SHARE/artifacts
```
(`-L` разыменовывает симлинк `last` → реальный каталог шага.)
