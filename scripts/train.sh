#!/usr/bin/env bash
# Train: hypernetwork (text or text+vision conditioning) or the stock lerobot
# PEFT-LoRA baseline, on ALL of lerobot/libero (40 tasks, 1693 episodes).
# The SmolVLA base stays frozen; in HN modes only the hypernet trains, in
# `lora` mode only the PEFT adapters do.
#
#   bash scripts/train.sh                       # vision-conditioned hypernet
#   MODE=text bash scripts/train.sh             # text-only hypernet (ablation)
#   MODE=lora bash scripts/train.sh             # stock PEFT LoRA, same sites
#   MODE=traj bash scripts/train.sh             # trajectory-conditioned hypernet
#   RESUME=1 bash scripts/train.sh              # resume OUTPUT from last ckpt
#
# Env vars:
#   MODE       (vision)               vision | text | lora | traj
#   RESUME     (0)                    1 = resume $OUTPUT from its last checkpoint
#   STEPS      (100000)               training steps
#   SEED       (42)                   training seed; non-default seeds get a
#                                     suffixed OUTPUT dir automatically
#   AUG        (0)                    1 = enable dataset image augmentations
#   EXPERT     (0)                    1 = also unfreeze + train the action
#                                     expert (HN modes only; checkpoint grows)
#   RANK       (4)                    LoRA rank (HN generation / PEFT adapter)
#   BATCH      (16 vision, 32 other)  train batch size
#   WORKERS    (8)                    dataloader workers
#   SAVE_FREQ  (25000)                checkpoint interval (steps)
#   OUTPUT     (outputs/<mode>…)      output dir (must not pre-exist unless RESUME=1)
#   PREC       (bf16)                 bf16 | no (fp32)
#   DINO_ID    (facebook/dinov2-base) external vision encoder (vision mode)
#   ENC        (dino)                traj clip encoder: dino | vjepa2
#   XPAIR_CACHE (outputs/xpair_cache/$ENC)  traj-clip cache dir (traj mode; build
#                                     it first with scripts/build_xpair_cache.py
#                                     --encoder $ENC)
#   KV         (0)                    1 = also inject LoRA at the VLM k/v routing
#                                     site (traj mode)
#   PAIR       (cross)               train pairing sugar -> hn_p_self:
#                                     same = 1.0 (context from the imitated episode),
#                                     cross|loo = 0.0 (one other episode of the task);
#                                     vision only: obs = legacy no-bank conditioning.
#   P_SELF     ()                    explicit hn_p_self override (e.g. 0.5 mix)
#   K          (1)                   context size (demos/frames per sample)
#   VGRID      (2)                   traj+vjepa2: s×s spatial tokens per tubelet
#   TSEL       (all)                 traj: temporal scope of a cached record —
#                                     all | first (minimal-clip arm)
#   TNF        (0)                   traj, TSEL=first: frames kept per record
#   TFILL      ()                    traj, TSEL=first: dup | zero (how the short
#                                     clip was padded to the encoder minimum)
#   TENC_MODEL ()                    traj: pin the HF encoder id the cache was
#                                     built with (empty = don't check)
#   TCHUNK     (0)                   traj: pin the encode window of the cache
#                                     (empty/0 = don't check)
#   DPATCH     (0)                   traj+dino: 1 = keep ALL 256 patch tokens of
#                                     every frame, unpooled (257 tokens/frame)
#   DGRID      (0)                   traj+dino: block-average the 16x16 patch map
#                                     to s×s tokens. DPATCH=1 is the s=16 case,
#                                     where the block is 1x1 and pooling is a no-op.
#   DREG       (0)                   traj+dino: register tokens per frame
#                                     (dinov2-with-registers checkpoints only)
#   TPOS       (none)                traj: temporal position code for the clip —
#                                     none | index | phase
#   TSUB       (0)                   traj: 1 = also embed the token's index WITHIN
#                                     its frame/tubelet (needed to interpret a null
#                                     result on the patch/register arms)
#   DTOK       (all)                 vision: DINO tokens fed to the HN — all | cls
#   TRUNK      ()                    traj: HF id of a pretrained trunk (e.g.
#                                     Qwen/Qwen3.5-0.8B) that replaces the scratch
#                                     hypernetwork — frozen Qwen3.5 text stack over
#                                     CACHED video tokens + instruction + 32 learned
#                                     layer tokens. Requires the qwen35vl cache
#                                     (build_qwen35_video_cache.py). Empty = off.
#   TSTRIDE    (4)                   trunk: keep every k-th frame of the demo (a
#                                     FIXED interval — same temporal resolution for
#                                     every episode; token count scales with length)
#   TEXT       (1)                   trunk: 1 = include the instruction in the trunk
#                                     input, 0 = video tokens only
#   BANK       (outputs/frame_bank.npz)  vision mode, PAIR=same|cross: first-frame
#                                     bank path (build with the frame-bank script)
#   VLM        (1)                   1 = also condition the HN on the VLM's own
#                                     image tokens (SigLIP of the frozen SmolVLM).
#                                     Applies to BOTH vision and traj modes. In traj
#                                     it requires $BANK: the conditioning frame must
#                                     be a t=0 bank frame on train, or the stream
#                                     would see a random step on train and t=0 at
#                                     eval. Historical default: ON in vision, absent
#                                     in traj — hence the _vlm / _novlm suffixes.
#   WANDB      (1)                    1 = enable wandb logging
#   WANDB_PROJECT (hyper-lora)        wandb project name
#   WANDB_OFFLINE (0)                 1 = log wandb locally without network/login
#                                     (later: `wandb sync wandb/offline-*`)
#   TB         (0)                    1 = log scalars to TensorBoard instead of wandb
#                                     (no network/login; view with
#                                     `tensorboard --logdir $OUTPUT/tensorboard`)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source venv/bin/activate

MODE="${MODE:-vision}"
RESUME="${RESUME:-0}"
STEPS="${STEPS:-100000}"
SEED="${SEED:-42}"
AUG="${AUG:-0}"
EXPERT="${EXPERT:-0}"
RANK="${RANK:-4}"
WORKERS="${WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-25000}"
PREC="${PREC:-bf16}"
DINO_ID="${DINO_ID:-facebook/dinov2-base}"
ENC="${ENC:-dino}"
XPAIR_CACHE="${XPAIR_CACHE:-outputs/xpair_cache/$ENC}"
KV="${KV:-0}"
PAIR="${PAIR:-cross}"
P_SELF="${P_SELF:-}"
K="${K:-1}"
VGRID="${VGRID:-2}"
TSEL="${TSEL:-all}"
TNF="${TNF:-0}"
TFILL="${TFILL:-}"
TENC_MODEL="${TENC_MODEL:-}"
TCHUNK="${TCHUNK:-0}"
DPATCH="${DPATCH:-0}"
DGRID="${DGRID:-0}"
[ "$DPATCH" = "1" ] && DGRID=16
DREG="${DREG:-0}"
TPOS="${TPOS:-none}"
TSUB="${TSUB:-0}"
DTOK="${DTOK:-all}"
TRUNK="${TRUNK:-}"
TSTRIDE="${TSTRIDE:-4}"
TEXT="${TEXT:-1}"
# --- overfit / training-extension knobs (direction 6) ---------------------------
# EPISODES: train on ONLY these dataset episodes (single-task overfit). Format is a
# python list, e.g. EPISODES="[57,58,59]"; empty = whole dataset (the old behaviour,
# byte-identical argv). Build the list for a task with scripts/pick_task.py.
# NO quotes inside the value: draccus rejects '"[57]"' for a list[int] field.
EPISODES="${EPISODES:-}"
# SCHED_DECAY: the SmolVLA lr preset floors at scheduler_decay_lr (2.5e-6) after
# scheduler_decay_steps=30k and does NOT scale with --steps, so steps 30k-100k of
# every run so far trained at 2.5% of peak lr. Set this when EXTENDING a run
# (STEPS > its trained steps) so the cosine spans the new horizon instead.
# Empty = don't touch the scheduler (the old behaviour).
SCHED_DECAY="${SCHED_DECAY:-}"
# PAIR sugar -> p_self (explicit P_SELF wins); "loo" is an alias of "cross".
case "$PAIR" in loo) PAIR=cross ;; esac
# These four now feed the output DIRECTORY NAME, so a typo would silently create a
# new arm instead of failing. Validate before anything uses them.
case "$PAIR"  in same|cross|obs) ;; *) echo "ERROR: PAIR must be same|cross|loo|obs, got '$PAIR'" >&2; exit 1 ;; esac
case "$TSEL"  in all|first)      ;; *) echo "ERROR: TSEL must be all|first, got '$TSEL'" >&2; exit 1 ;; esac
case "$TFILL" in ""|dup|zero)    ;; *) echo "ERROR: TFILL must be empty|dup|zero, got '$TFILL'" >&2; exit 1 ;; esac
case "$TPOS"  in none|index|phase) ;; *) echo "ERROR: TPOS must be none|index|phase, got '$TPOS'" >&2; exit 1 ;; esac
case "$DTOK"  in all|cls)        ;; *) echo "ERROR: DTOK must be all|cls, got '$DTOK'" >&2; exit 1 ;; esac
if [ "$TSEL" = "first" ] && [ "$TNF" -lt 1 ]; then
    echo "ERROR: TSEL=first requires TNF>=1, got '$TNF'" >&2; exit 1
fi
if [ -z "$P_SELF" ]; then
    case "$PAIR" in
        same)  P_SELF=1.0 ;;
        cross) P_SELF=0.0 ;;
        obs)   P_SELF=1.0 ;;   # unused without a bank; vision legacy mode
    esac
fi
BANK="${BANK:-outputs/frame_bank.npz}"
VLM="${VLM:-1}"
[ "$KV" = "1" ] && KV_FLAG=true || KV_FLAG=false
export ACCELERATE_MIXED_PRECISION="$PREC"
[ "${WANDB:-1}" = "1" ] && WANDB_FLAG=true || WANDB_FLAG=false
[ "$AUG" = "1" ] && AUG_FLAG=true || AUG_FLAG=false
[ "$EXPERT" = "1" ] && EXPERT_FLAG=true || EXPERT_FLAG=false
WANDB_PROJECT="${WANDB_PROJECT:-hyper-lora}"
[ "${WANDB_OFFLINE:-0}" = "1" ] && export WANDB_MODE=offline
# TB=1 substitutes the logger class inside train_hyper_lora.py; the wandb enable
# flag must be true so the train loop instantiates a logger at all.
[ "${TB:-0}" = "1" ] && { WANDB_FLAG=true; export TENSORBOARD=1; }

case "$MODE" in
    vision) DEFAULT_OUTPUT="outputs/hyper_lora_vision"; BATCH="${BATCH:-16}" ;;
    text)   DEFAULT_OUTPUT="outputs/hyper_lora_text";   BATCH="${BATCH:-32}" ;;
    lora)   DEFAULT_OUTPUT="outputs/lora_baseline";     BATCH="${BATCH:-32}" ;;
    traj)   DEFAULT_OUTPUT="outputs/hyper_lora_traj_$ENC"; BATCH="${BATCH:-32}" ;;
    *) echo "ERROR: MODE must be vision | text | lora | traj, got '$MODE'" >&2; exit 1 ;;
esac
# Ablation toggles get distinct default output dirs so runs don't collide.
[ "$RANK" != "4" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_r${RANK}"
[ "$SEED" != "42" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_s${SEED}"
[ "$AUG" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_aug"
[ "$EXPERT" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_expert"
[ "$MODE" = "traj" ] && [ "$KV" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_kv"
# CONDITIONING knobs must appear in the name too. Without this, arms that differ
# only by pairing / context size / token format map to the SAME default dir — the
# run then either aborts on the pre-exist check below or, with RESUME=1, writes
# into a different arm's checkpoint. (This already collided B1<->B2 and A3<->A4.)
# A suffix is added ONLY when a knob is off its default, exactly like the
# RANK/SEED/AUG lines above — so the fully default invocation of each mode keeps
# its historical path and RESUME=1 still finds those checkpoints. A run that used
# a NON-default knob before this change did share a path with its siblings; to
# resume one of those, pass OUTPUT=… explicitly.
case "$MODE" in
    vision)
        if [ "$PAIR" = "obs" ]; then DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_obs"
        elif [ "$P_SELF" != "0.0" ]; then DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_p${P_SELF}"; fi
        [ "$VLM" = "0" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_novlm"
        [ "$DTOK" != "all" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_${DTOK}" ;;
    traj)
        [ "$P_SELF" != "0.0" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_p${P_SELF}"
        [ "$K" != "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_k${K}"
        [ "$ENC" = "vjepa2" ] && [ "$VGRID" != "2" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_g${VGRID}"
        [ "$TSEL" != "all" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_${TSEL}${TNF}${TFILL}"
        [ "$DGRID" = "16" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_patches"
        [ "$DGRID" != "0" ] && [ "$DGRID" != "16" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_dg${DGRID}"
        [ "$DREG" != "0" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_reg${DREG}"
        [ "$TPOS" != "none" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_${TPOS}"
        [ "$TSUB" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_sub"
        [ "$VLM" = "1" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_vlm"
        if [ -n "$TRUNK" ]; then
            short="$(basename "$TRUNK" | tr 'A-Z' 'a-z')"
            DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_trunk_${short}_e${TSTRIDE}"
            [ "$TEXT" = "0" ] && DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_notext"
        fi
        # A different cache IS a different arm (TENC_MODEL/TCHUNK only *assert*
        # provenance; XPAIR_CACHE is what actually selects the data). Skipped when
        # the flags above already spell the cache name out.
        if [ "$XPAIR_CACHE" != "outputs/xpair_cache/$ENC" ]; then
            CACHE_TAG="$(basename "$XPAIR_CACHE")"
            case "$DEFAULT_OUTPUT" in
                *"$CACHE_TAG"*) : ;;
                *) DEFAULT_OUTPUT="${DEFAULT_OUTPUT}_${CACHE_TAG}" ;;
            esac
        fi ;;
esac
OUTPUT="${OUTPUT:-$DEFAULT_OUTPUT}"

if [ "$RESUME" = "1" ]; then
    CONFIG="$OUTPUT/checkpoints/last/pretrained_model/train_config.json"
    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: $CONFIG not found — nothing to resume." >&2
        exit 1
    fi
    # lerobot reads `steps` from the SAVED config on resume (the loop is
    # range(saved_step, cfg.steps)), so plain RESUME=1 cannot extend a run: it would
    # "finish" instantly at its old step count. Pass --steps ONLY when the caller
    # asks for a different horizon — otherwise the argv stays exactly as before.
    RESUME_ARGS=()
    CFG_STEPS="$(python -c "import json;print(json.load(open('$CONFIG')).get('steps'))" 2>/dev/null || echo "?")"
    if [ "$STEPS" != "$CFG_STEPS" ]; then
        RESUME_ARGS+=(--steps="$STEPS")
        if [ -z "$SCHED_DECAY" ]; then
            echo "WARNING: extending $CFG_STEPS -> $STEPS steps WITHOUT SCHED_DECAY." >&2
            echo "         The lr preset floors at 2.5e-6 after its decay horizon; the" 2>&1
            echo "         extension would train at a dead lr. Recommended:" >&2
            echo "         SCHED_DECAY=$((STEPS * 3 / 10)) STEPS=$STEPS RESUME=1 ..." >&2
        fi
    fi
    [ -n "$SCHED_DECAY" ] && RESUME_ARGS+=(--policy.scheduler_decay_steps="$SCHED_DECAY")
    echo "==> Resuming $OUTPUT from $(readlink -f "$OUTPUT/checkpoints/last") | save_freq=$SAVE_FREQ | steps=$STEPS (config: $CFG_STEPS)"
    # shellcheck disable=SC2068
    exec python train_hyper_lora.py --config_path="$CONFIG" --resume=true --save_freq="$SAVE_FREQ" ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
fi

if [ -e "$OUTPUT" ]; then
    echo "ERROR: output dir '$OUTPUT' already exists (lerobot refuses to overwrite)."
    echo "       Set OUTPUT=… to a fresh path, remove it, or pass RESUME=1."
    exit 1
fi

# Work around lerobot/libero's wrong meta/episodes file_index (idempotent, ~21MB):
# pre-fetch every data parquet so the loader globs + filters by episode_index.
python -c "from src.data.libero import prefetch_all_data_parquets as p; p()"

# Mode-specific policy flags.
MODE_ARGS=()
case "$MODE" in
    vision)
        # PAIR=obs -> legacy conditioning on the current observation (no bank).
        # PAIR=same|cross -> the t=0 bank frame with hn_p_self (1.0 = own episode,
        # 0.0 = one random other episode of the task), resampled every step.
        MODE_ARGS+=(
            --policy.type=hyper_lora_smolvla
            --policy.hn_use_vlm_vision=$([ "$VLM" = "1" ] && echo true || echo false)
            --policy.hn_use_dino=true
            --policy.hn_dino_model_id="$DINO_ID"
            --policy.hn_dino_tokens="$DTOK"
            --policy.lora_rank="$RANK" --policy.lora_alpha=$((RANK * 4))
            --policy.train_action_expert="$EXPERT_FLAG"
        )
        [ "$PAIR" != "obs" ] && MODE_ARGS+=(
            --policy.hn_frame_bank_path="$BANK"
            --policy.hn_p_self="$P_SELF"
        ) ;;
    text)
        MODE_ARGS+=(
            --policy.type=hyper_lora_smolvla
            --policy.lora_rank="$RANK" --policy.lora_alpha=$((RANK * 4))
            --policy.train_action_expert="$EXPERT_FLAG"
        ) ;;
    lora)
        # Stock lerobot PEFT LoRA on the same injection sites as the hypernet
        # (VLM text_model MLP linears). The trainer wraps the policy with PEFT
        # when --peft.* is given; the checkpoint saves a standard adapter that
        # lerobot-eval loads automatically (config.use_peft=true).
        MODE_ARGS+=(
            --policy.path=HuggingFaceVLA/smolvla_libero
            --peft.r="$RANK"
            --peft.target_modules='model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)'
        ) ;;
    traj)
        # Trajectory-conditioned hypernet: the HN reads a demo clip from the
        # offline cache. Build it first with scripts/build_xpair_cache.py.
        if [ ! -d "$XPAIR_CACHE" ]; then
            echo "ERROR: traj cache '$XPAIR_CACHE' not found." >&2
            case "$XPAIR_CACHE" in
                /*/*) : ;;
                /*)   echo "       That path starts at the filesystem root with a single" >&2
                      echo "       component — the usual cause is an UNSET shell variable," >&2
                      echo "       e.g. XPAIR_CACHE=\$D/dino with \$D empty. Pass the full" >&2
                      echo "       path literally: XPAIR_CACHE=outputs/xpair_cache/dino" >&2 ;;
            esac
            echo "       Available caches:" >&2
            ls -1d outputs/xpair_cache/*/ 2>/dev/null | sed 's|^|         |' >&2 \
                || echo "         (none — build them: bash scripts/build_all_caches.sh)" >&2
            exit 1
        fi
        if [ "$VLM" = "1" ] && [ ! -f "$BANK" ]; then
            echo "ERROR: VLM=1 in traj mode needs the t=0 frame bank '$BANK'." >&2
            echo "       Build it: python scripts/build_frame_bank.py --out $BANK" >&2
            echo "       (or run with VLM=0 to drop the VLM stream)" >&2
            exit 1
        fi
        MODE_ARGS+=(
            --policy.type=traj_hyper_lora_smolvla
            --policy.hn_use_traj_clip=true
            --policy.hn_xpair_cache_path="$XPAIR_CACHE"
            --policy.hn_traj_encoder="$ENC"
            --policy.hn_use_vlm_vision=$([ "$VLM" = "1" ] && echo true || echo false)
            --policy.hn_p_self="$P_SELF"
            --policy.hn_context_k="$K"
            --policy.hn_vjepa_grid="$VGRID"
            --policy.hn_inject_vlm_kv="$KV_FLAG"
            --policy.lora_rank="$RANK" --policy.lora_alpha=$((RANK * 4))
            --policy.train_action_expert="$EXPERT_FLAG"
        )
        # Appended only when actually used, so the command line of every existing
        # arm stays byte-identical and no empty-string flag reaches the parser.
        [ "$TSEL" != "all" ] && MODE_ARGS+=(
            --policy.hn_traj_time_select="$TSEL"
            --policy.hn_traj_n_frames="$TNF"
        )
        [ -n "$TFILL" ] && MODE_ARGS+=(--policy.hn_traj_fill="$TFILL")
        [ -n "$TENC_MODEL" ] && MODE_ARGS+=(--policy.hn_traj_encoder_model="$TENC_MODEL")
        [ "$TCHUNK" != "0" ] && MODE_ARGS+=(--policy.hn_traj_chunk="$TCHUNK")
        [ "$DGRID" != "0" ] && MODE_ARGS+=(--policy.hn_dino_grid="$DGRID")
        [ "$DREG" != "0" ] && MODE_ARGS+=(--policy.hn_dino_n_reg="$DREG")
        [ "$TPOS" != "none" ] && MODE_ARGS+=(--policy.hn_traj_pos_emb="$TPOS")
        [ "$TSUB" = "1" ] && MODE_ARGS+=(--policy.hn_traj_sub_emb=true)
        # The bank is what keeps the VLM stream's train frame (t=0 of some episode
        # of the task) in the same distribution as its eval frame (t=0 of the
        # rollout). Without it the arm measures a train/eval mismatch instead.
        [ "$VLM" = "1" ] && MODE_ARGS+=(--policy.hn_frame_bank_path="$BANK")
        if [ -n "$TRUNK" ]; then
            MODE_ARGS+=(
                --policy.hn_trunk_model="$TRUNK"
                --policy.hn_trunk_stride="$TSTRIDE"
                --policy.hn_trunk_text="$([ "$TEXT" = "1" ] && echo true || echo false)"
            )
        fi
        : ;;   # the `:` keeps the branch's exit status 0 under `set -e`
esac

echo "==> Train | mode=$MODE | rank=$RANK | prec=$PREC | batch=$BATCH | seed=$SEED | aug=$AUG_FLAG | expert=$EXPERT_FLAG | wandb=$WANDB_FLAG"
echo "    output=$OUTPUT"
python train_hyper_lora.py \
    "${MODE_ARGS[@]}" \
    --dataset.repo_id=lerobot/libero \
    --dataset.use_imagenet_stats=false \
    --dataset.image_transforms.enable="$AUG_FLAG" \
    ${EPISODES:+--dataset.episodes=$EPISODES} \
    ${SCHED_DECAY:+--policy.scheduler_decay_steps=$SCHED_DECAY} \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --steps="$STEPS" \
    --batch_size="$BATCH" \
    --num_workers="$WORKERS" \
    --save_freq="$SAVE_FREQ" \
    --save_checkpoint=true \
    --seed="$SEED" \
    --wandb.enable="$WANDB_FLAG" \
    --wandb.project="$WANDB_PROJECT" \
    --output_dir="$OUTPUT"

echo
echo "Done. Checkpoint: $OUTPUT/checkpoints/last/pretrained_model"
echo "Eval it with:  POLICIES=\"$MODE=$OUTPUT/checkpoints/last/pretrained_model\" bash scripts/eval.sh"
