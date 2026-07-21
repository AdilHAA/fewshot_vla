# Hypernetwork-LoRA для SmolVLA: адаптация VLA-политики по условию задачи

Гиперсеть (HN), которая **за один forward генерирует LoRA-адаптер** для замороженной
VLA-политики SmolVLA по условию задачи — тексту инструкции, кадру наблюдения или
демонстрациям задачи. На тесте адаптация не требует ни одного градиентного шага:
условие → гиперсеть → веса адаптера → политика действует.

Идея восходит к Hyper-LoRA (Sharma, Xiong et al., «Efficient Domain Adaptation of
Robotic Foundation Models via Hypernetwork-Generated LoRA», NeurIPS 2024 FT Workshop):
трансформер-бэкбон заморожен, HN генерирует LoRA на MLP-слои, голова действий
доучивается. Здесь этот рецепт перенесён с Octo на SmolVLA и расширен
кондиционированием на кадры и демонстрации, оценкой на OOD-осях LIBERO-Pro и
sim-аугментациями демонстраций.

---

## 1. Как это работает (5 минут на вход)

```
условие ──────────────► HyperNetwork ─────────► LoRA ΔW ──────► SmolVLA
(текст / кадр / демо)   concat self-attention   r=4, α=16       VLM заморожен,
                        [32 layer-query |       32 слоя ×       ΔW вставляется в MLP;
                         текст | условие]       {gate,up,down}  action expert обучается
                                                                (flow matching → действия)
```

- **Вход HN** — одна последовательность токенов: 32 обучаемых *layer-query*
  (по одному на слой VLM) + ≤48 токенов инструкции (токенайзер и embedding-слой
  самого VLM, затем обучаемая проекция в d=128) + токены условия (зависят от режима).
- **Readout** — после self-attention берутся первые 32 позиции; токен слоя *i*
  проходит через две **общие для всех слоёв** линейные головы:
  `head_down: Linear(128 → r·in)` и `head_up: Linear(128 → out·r)` (zero-init ⇒
  на старте ΔW=0 и политика совпадает с базовой). Reshape — и готовы матрицы
  `W_down (4×in)`, `W_up (out×4)` для каждого из 32 слоёв × 3 проекций.
- **Вставка** — `src/hyper_lora/dynamic_lora.py` оборачивает `nn.Linear` VLM:
  `y = Wx + (α/r)·W_up·W_down·x`; сгенерированные веса подставляются перед forward
  и не являются параметрами модуля.
- **Обучение** — чистый flow-matching лосс политики на `lerobot/libero`
  (40 задач, ~1.7k демо). Учатся: гиперсеть (+проекции) и action expert
  (`EXPERT=1`, как в исходной статье). VLM и vision-энкодеры заморожены всегда.
- **Eval** — LIBERO (4 сьюта, in-distribution) + LIBERO-Pro (оси возмущений
  `_lan`/`_object`/`_swap`/`_task` поверх LIBERO-10). С `EPISODE_CACHE=1` адаптер
  строится на первом шаге эпизода и замораживается до его конца.

### Режимы кондиционирования (что именно видит гиперсеть)

| MODE | условие HN | политика (тип) |
|---|---|---|
| `text` | только инструкция | `hyper_lora_smolvla` |
| `vision` | инструкция + кадр: SigLIP-токены VLM и/или 257 патчей DINOv2; источник кадра — ручка `PAIR` (см. §5) | `hyper_lora_smolvla` |
| `traj` | инструкция + **целые траектории демо** (CLS-токен на кадр у DINO / mean-pool на tubelet у V-JEPA2), напрямую в attention без пулинга | `traj_hyper_lora_smolvla` |
| `lora` | — (базлайн: обычный PEFT-LoRA, градиентный) | стоковый lerobot |

---

## 2. Карта репозитория

```
train_hyper_lora.py        тренинг: обёртка над lerobot-train; регистрирует политики,
                           подгоняет SmolVLA-конфиг под базовый чекпоинт, TB-логгер
eval_hyper_lora.py         eval: регистрирует политики + LIBERO-Pro сьюты, отдаёт lerobot-eval
eval_libero_pro.py         то же для базовых чекпоинтов без гиперсети

src/hyper_lora/            ПОЛИТИКА №1: text/vision-кондиционирование
  hypernetwork.py            ядро HN: стримы → self-attention → shared-головы весов
  dynamic_lora.py            обёртка Linear с подставными LoRA-весами
  modeling_hyper_lora_smolvla.py   lerobot-glue: патчинг VLM, инжект, frame-bank режимы
  configuration_...py        конфиг (все hn_* ручки), base_config.py — сверка с базой

src/hyper_lora_traj/       ПОЛИТИКА №2 (наследует №1): демо-кондиционирование
  fusion_hypernetwork.py     HN + 4-й стрим: сырые токены траекторий + маркеры границ демо
  modeling_...py             трейн: leave-one-out селектор; eval: резолв задачи → K демо из кеша
  configuration_...py        ручки hn_pair_mode / hn_context_k / hn_xpair_cache_path

src/traj_data/             ЧИСТЫЕ модули без lerobot (данные условия)
  encoder.py                 DINOv2 (CLS/кадр) и V-JEPA2 (mean-pool/tubelet) энкодеры клипов
  cache_io.py / traj_cache.py  ragged-кеш эмбеддингов: tokens.mmap + index.json (offset/length),
                             lookup задач по task_index и по тексту инструкции (fuzzy)
  xpair_select.py            селекторы контекста: train same|loo, eval K демо; паддинг+маски
  frame_bank.py              банк стартовых кадров (t=0) для vision-абляций same/cross
  augment.py / diagnostics.py  вспомогательные (индексы клипов; измерительные метрики)

src/libero_pro/            LIBERO-Pro поверх родного LIBERO: register.py (сьюты),
                           objects.py (синтез перекрашенных объектов), fetch.py
src/data/libero.py         обход бага file_index в lerobot/libero (префетч паркетов)

scripts/
  setup.sh                   бутстрап окружения с нуля (venv + пины + LIBERO)
  train.sh / eval.sh         единые точки запуска (все ручки — env-переменными, см. §4–5)
  render_recolor_clips.py    sim-аугментации демо: MuJoCo state-replay с перекраской
                             объектов и camera-jitter (без физики — кадры валидного успеха)
  build_xpair_cache.py       оффлайн-кеш эмбеддингов всех демо (все кадры, все варианты)
  build_frame_bank.py        банк t=0 кадров (+ ауг-варианты + тексты задач)
  patch_lerobot.py           пост-инсталл фикс lerobot 0.5.1 под transformers 5 / py3.12
  analyze_lora.py            зонд: зависят ли сгенерированные LoRA от задачи
```

Договорённость по коду: **`src/traj_data/` не импортирует lerobot** (чистые
numpy/torch-модули с узкими интерфейсами); всё lerobot-связанное живёт в
`modeling_*`/скриптах. Скрипты держат «чистую» логику в верхнеуровневых функциях,
а lerobot/GPU-код — в `main()`.

---

## 3. Установка

```bash
git clone <repo> && cd fewshot_vla
bash scripts/setup.sh          # venv + зависимости + LIBERO; дальше ничего руками
```

Критичные пины (не обновлять бездумно — каждый закреплён после реального падения):

| пакет | пин | почему |
|---|---|---|
| `lerobot` | `==0.5.1` | 0.6+ ломается на новом PyAV и меняет dataset-API |
| `transformers` | `==5.7.0` | 5.13 валит импорт lerobot (groot) и дрейфит токенизацию SmolVLM |
| `mujoco` | `<3.3` | 3.3 сменил сигнатуру `mj_fullM`, несовместим с robosuite 1.4 |

Ещё грабли: `HF_HUB_OFFLINE=1` использовать **нельзя** (lerobot 0.5.1 всегда ходит
в hub за списком ревизий датасета); headless-рендер — через EGL
(`MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=<gpu>` — выставляется в скриптах).

---

## 4. Данные и подготовка (для traj-режима и vision-абляций)

Датасет `lerobot/libero` скачивается автоматически при первом запуске. Дальше —
только если нужны демо-кондиционирование или sim-аугментации:

```bash
# 1. HDF5-демо LIBERO (содержат сырые MuJoCo-состояния; официальные ссылки мертвы —
#    качаем HF-зеркало):
hf download yifengzhu-hf/LIBERO-datasets --repo-type dataset \
    --local-dir third_party/LIBERO/libero/datasets \
    --include "libero_10/*" "libero_goal/*" "libero_object/*" "libero_spatial/*"

# 2. Sim-аугментации: replay записанных состояний в перекрашенной сцене / со сдвинутой
#    камерой -> те же успешные движения, другой внешний вид. Резюмируемо, шардируется:
python scripts/render_recolor_clips.py --colors red,green --cam_jitters 2 --all_frames \
    --gate_mae 0.22 --out outputs/rendered_recolor
#    (--shard i --num_shards n для параллельных карт; проверяй gate mae в логе и contact_sheet)

# 3. Кеши эмбеддингов демо (encode-once: все кадры всех эпизодов + ауг-варианты):
python scripts/build_xpair_cache.py --encoder dino   --rendered_dir outputs/rendered_recolor --out outputs/xpair_cache/dino
python scripts/build_xpair_cache.py --encoder vjepa2 --rendered_dir outputs/rendered_recolor --out outputs/xpair_cache/vjepa2

# 4. Банк стартовых кадров (vision-абляции):
python scripts/build_frame_bank.py --rendered_dir outputs/rendered_recolor --out outputs/frame_bank.npz
```

Про gate: рендер сверяется с реальным кадром датасета (MAE) — это защита от
рассинхрона сцены/камеры; задачи, не прошедшие порог, пропускаются с warning.

---

## 5. Обучение

Всё через `scripts/train.sh`, конфигурация — env-переменными:

```bash
# text / vision / PEFT-базлайн
MODE=text   EXPERT=1 AUG=1 bash scripts/train.sh
MODE=vision EXPERT=1 AUG=1 bash scripts/train.sh
MODE=lora   bash scripts/train.sh

# демо-кондиционирование (leave-one-out по умолчанию)
MODE=traj ENC=dino   PAIR=loo  EXPERT=1 AUG=1 OUTPUT=outputs/dino_loo  bash scripts/train.sh
MODE=traj ENC=vjepa2 PAIR=same EXPERT=1 AUG=1 OUTPUT=outputs/jepa_same bash scripts/train.sh
```

Главные ручки (полный список — в шапке train.sh):

| ручка | значения | что делает |
|---|---|---|
| `MODE` | text/vision/lora/traj | режим кондиционирования |
| `EXPERT` | 0/**1** | обучать ли action expert (1 = как в исходной статье) |
| `AUG` | 0/1 | встроенные lerobot-аугментации картинок на входе политики |
| `PAIR` | traj: `same`\|`loo`; vision: `obs`\|`same`\|`cross` | **протокол обучения** — откуда контекст HN (см. ниже) |
| `ENC` | dino/vjepa2 | энкодер демо (traj) |
| `K` | 8 | размер контекста (демо/кадров) |
| `VLM` | 0/1 | vision: включать ли SigLIP-стрим VLM (0 = только DINO) |
| `TB` / `WANDB_OFFLINE` | 0/1 | TensorBoard в `$OUTPUT/tensorboard` / оффлайн-wandb |
| `OUTPUT`, `STEPS`, `BATCH`, `SEED`, `RANK` | … | стандартные |

### Протокол PAIR — центральная абляция проекта

Вопрос: чему учится HN, если контекст берётся из *той же* или из *других*
траекторий задачи? Траектории одной задачи различаются только «рандомом»
инициализации (позиции, дистракторы), поэтому чужой контекст заставляет HN
извлекать инвариант задачи вместо копирования сцены.

- **traj**: `same` — контекст = сам имитируемый эпизод (целиком);
  `loo` — 8 демо той же задачи *без* имитируемого (leave-one-out, пересэмпл каждый шаг).
- **vision**: `obs` — кадр текущего шага (легаси-режим);
  `same` — t=0 кадр своего эпизода из банка;
  `cross` — 8 t=0 кадров *разных других* эпизодов задачи.
- На eval контекст всегда согласован с трейн-форматом: traj берёт K демо задачи из
  кеша (задача резолвится по task_index или fuzzy-матчем инструкции), vision-cross —
  K кадров train-эпизодов из банка; same-модели получают контекст размера 1.

---

## 6. Оценка

`scripts/eval.sh` — матрица policies × suites × seeds, резюмируемая (готовые ячейки
пропускаются), со сводной таблицей в конце:

```bash
EPISODE_CACHE=1 SEEDS=1000 \
TASKS="libero_10 libero_goal libero_object libero_spatial libero_10_lan libero_10_object libero_10_swap libero_10_task" \
POLICIES="my_run=outputs/my_run/checkpoints/last/pretrained_model base=HuggingFaceVLA/smolvla_libero" \
bash scripts/eval.sh
```

- `EPISODE_CACHE=1` — адаптер строится один раз на t=0 эпизода и замораживается.
  **Обязателен для traj-политик**, нужен и для vision-абляций.
- Результаты: `outputs/eval_matrix/<label>/<suite>/seed_<s>/eval_info.json`
  (`overall.pc_success` — success rate). Для финальных чисел — ≥3 сида.
- Диагностика traj-eval: в логе строки `[TRAJ] task=<id> demos=<n> tokens=<n>` —
  подтверждение, что задача разрезолвилась и контекст подтянулся из кеша.
- Зонды: `HN_LOG_LORA=1` (норма/дрейф генерируемых LoRA), `HN_LOG_ACTION=1`
  (норма/джерк действий).

### Ориентиры (seed 1000, 50 эп/задачу, success rate %)

| метод | LIBERO-10 | goal | object | spatial | _lan | _object | _swap | _task |
|---|---|---|---|---|---|---|---|---|
| base (frozen SmolVLA) | 41.8 | 84.2 | 91.0 | 77.4 | 0.0 | 18.8 | 0.0 | 0.0 |
| PEFT-LoRA r=8 | 43.6 | 85.0 | 85.4 | 81.4 | 0.0 | 14.8 | 0.0 | 0.0 |
| HN text | 59.0 | 90.6 | 82.6 | 87.0 | 3.8 | 28.0 | 0.0 | 0.6 |
| HN vision | 61.0 | 93.4 | 85.4 | 87.8 | 3.6 | 28.2 | 0.0 | 0.6 |

Гиперсеть даёт +17–20 п.п. in-distribution против базы и PEFT-LoRA; условие
«текст/кадр» почти не переносится на OOD-оси — это и мотивирует демо-кондиционирование
(traj) и протокольные абляции (PAIR). `_swap`/`_task` — пространственный grounding,
вне досягаемости кондиционирования по построению.

---

## 7. Частые вопросы при онбординге

- **Почему два класса политик?** `traj_hyper_lora_smolvla` наследует
  `hyper_lora_smolvla` и с выключенными флагами структурно идентичен ему —
  чекпоинты и код text/vision-режимов не затронуты traj-стеком.
- **Почему кеш ragged?** Демо разной длины; храним конкат токенов одним mmap +
  (offset, length) на запись — ленивое чтение без паддинга на диске.
- **Почему у HN нет своего токенайзера/эмбеддера текста?** Инструкцию токенизирует
  препроцессор политики, эмбеддит замороженная матрица VLM — HN видит текст теми же
  векторами, что и политика, и учит только проекцию 960→128.
- **wandb недоступен с кластера?** `TB=1` (TensorBoard, без сети и логина) или
  `WANDB_OFFLINE=1` (`wandb sync` позже). Лосс дублируется в stdout — снимайте через
  `... bash scripts/train.sh 2>&1 | tee train.log`.
- **Что-то висит после «Creating policy»?** Проверьте `nvidia-smi` (lerobot молчит
  до первого log-шага) и стек через `py-spy dump` — типичные виновники: wandb без
  сети, гонка HF-кеша при параллельных стартах (запускайте со сдвигом ~60 с).
