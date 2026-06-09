#!/usr/bin/env bash
# One-shot, non-interactive bootstrap for a fresh machine.
#
# Goal: `git clone … && bash scripts/setup.sh` yields a fully working env
# (training + LIBERO/LIBERO-Pro eval) with no further manual steps.
#
# Idempotent: safe to re-run. Repairs a broken/foreign pre-existing venv,
# installs deps, patches upstream lerobot, initializes the LIBERO config
# non-interactively, and verifies torch+CUDA / lerobot / libero / mujoco.
#
# Usage:
#     bash scripts/setup.sh
#
# Env vars:
#     PYTHON_BIN       (default: python3)             Python interpreter to use
#     VENV_DIR         (default: venv)                Where to create the venv
#     PREFETCH_MODELS  (default: 0; set 1 to enable)  Download base + VLM (~3GB)
#     PREFETCH_DATA    (default: 0; set 1 to enable)  Download lerobot/libero (~2GB)
#     PREFETCH         (default: unset)               'all' = both flags above
#     SKIP_SIM         (default: 0; set 1 to enable)  Skip LIBERO simulator deps
#                                                     (faster, training-only)

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-venv}"
REQ_FILE="${REQ_FILE:-requirements.txt}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo:    $REPO_ROOT"
echo "==> Python:  $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
echo "==> Venv:    $VENV_DIR"

# 1. Python version check (lerobot needs >= 3.10)
if ! "$PYTHON_BIN" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "ERROR: Python 3.10+ is required (lerobot dependency)."
    exit 1
fi

# 2. Create venv if missing; repair if a pre-existing one is broken.
#    (A venv copied/moved between machines can have a dangling interpreter
#    symlink — `bin/python` points at a base prefix that no longer exists.)
venv_ok() { [ -x "$VENV_DIR/bin/python" ] && "$VENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1; }

if [ -d "$VENV_DIR" ] && ! venv_ok; then
    echo "==> Existing venv at $VENV_DIR is broken (dead interpreter); recreating"
    rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtualenv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090,SC1091
source "$VENV_DIR/bin/activate"

if ! venv_ok; then
    echo "ERROR: venv interpreter still not usable after (re)creation: $VENV_DIR"
    exit 1
fi

# 3. Upgrade pip + install requirements
pip install --upgrade pip setuptools wheel

if [ "${SKIP_SIM:-0}" = "1" ]; then
    echo "==> Installing training-only deps (skipping LIBERO simulator)"
    # Strip LIBERO sim block (everything after the matching comment header)
    TMP_REQ="$(mktemp)"
    awk '/^# LIBERO simulator/ {exit} {print}' "$REQ_FILE" > "$TMP_REQ"
    pip install -r "$TMP_REQ"
    rm -f "$TMP_REQ"
else
    pip install -r "$REQ_FILE"
fi

# 3b. Patch upstream lerobot incompatibilities (idempotent).
#     Fixes GR00TN15Config @dataclass + transformers>=5 import failure.
echo "==> Patching lerobot for transformers>=5 compatibility"
python scripts/patch_lerobot.py

# 3b2. Install LIBERO from source (idempotent, sim builds only).
#      Upstream LIBERO has NO top-level `libero/__init__.py`, so
#      `pip install git+…` runs find_packages() → [] and builds an EMPTY wheel
#      (metadata only, no code) → `import libero` fails. Instead we clone it and
#      register the PEP-420 namespace package via a .pth on the venv path. Its
#      runtime deps (robosuite/bddl/mujoco/…) are pinned in requirements.txt.
if [ "${SKIP_SIM:-0}" != "1" ]; then
    LIBERO_DIR="$REPO_ROOT/third_party/LIBERO"
    if [ ! -d "$LIBERO_DIR/libero" ]; then
        echo "==> Cloning LIBERO into $LIBERO_DIR"
        mkdir -p "$REPO_ROOT/third_party"
        git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
    fi
    SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
    echo "$LIBERO_DIR" > "$SITE_PACKAGES/libero_local.pth"
    echo "==> Registered LIBERO: $SITE_PACKAGES/libero_local.pth -> $LIBERO_DIR"
fi

# 3c. Initialize the LIBERO config non-interactively (sim builds only).
#     `import libero.libero` calls input() on first import if
#     ~/.libero/config.yaml is missing → EOFError under any non-tty run.
#     Feeding "N" creates the default config (bddl/init/assets resolve into
#     the installed package; the unused raw-datasets path stays unset).
#     Idempotent: libero only prompts when the config is absent.
if [ "${SKIP_SIM:-0}" != "1" ]; then
    echo "==> Initializing LIBERO config (~/.libero/config.yaml)"
    printf 'N\n' | python -c "import libero.libero" >/dev/null 2>&1 || true
    if ! python -c "import os,sys; p=os.path.expanduser('~/.libero/config.yaml'); sys.exit(0 if os.path.exists(p) else 1)"; then
        echo "ERROR: failed to initialize ~/.libero/config.yaml" >&2
        echo "       Try: printf 'N\\n' | python -c 'import libero.libero'" >&2
        exit 1
    fi
    # Silence robosuite's 'no private macro file' warning (writes a copy).
    python "$(python -c 'import robosuite,os;print(os.path.join(os.path.dirname(robosuite.__file__),"scripts","setup_macros.py"))')" \
        >/dev/null 2>&1 || true
fi

# 4. Smoke-test imports
echo "==> Verifying core imports..."
SKIP_SIM="${SKIP_SIM:-0}" python - <<'PY'
import importlib, os, sys

core = ["torch", "lerobot", "transformers", "huggingface_hub", "wandb"]
if os.environ.get("SKIP_SIM", "0") != "1":
    core += ["libero.libero", "robosuite", "mujoco"]

failures = []
for mod in core:
    try:
        importlib.import_module(mod)
        print(f"  [OK]   {mod}")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {mod}: {e}")
        failures.append(mod)

try:
    import torch
    print(f"  [OK]   torch {torch.__version__} | CUDA={torch.cuda.is_available()} "
          f"| devices={torch.cuda.device_count()}")
except Exception as e:  # noqa: BLE001
    print(f"  [FAIL] torch/CUDA probe: {e}")
    failures.append("torch-cuda")

if failures:
    sys.exit(1)
PY

# 4b. Headless MuJoCo render check (non-fatal). Confirms the simulator can
#     render offscreen on this machine — the #1 thing that silently breaks
#     LIBERO eval. We pin MUJOCO_EGL_DEVICE_ID=0: plain egl enumerates ALL EGL
#     devices and can pick a non-renderable one (init succeeds but rendering
#     fails). The eval shims set the same two vars. Harmless EGL __del__ noise
#     during context teardown is suppressed (2>/dev/null) and does not affect
#     the render result.
if [ "${SKIP_SIM:-0}" != "1" ]; then
    echo "==> Checking headless MuJoCo render (MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0)..."
    if MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python - <<'PY' 2>/dev/null
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='sphere' size='0.1'/></worldbody></mujoco>")
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 64, 64)
mujoco.mj_forward(m, d)
r.update_scene(d)
assert r.render().shape == (64, 64, 3)
PY
    then
        echo "  [OK]   offscreen render works (MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0)"
    else
        echo "  [WARN] egl offscreen render failed. Eval needs a working GL"
        echo "         backend. Try a different MUJOCO_EGL_DEVICE_ID, or"
        echo "         MUJOCO_GL=osmesa (CPU), or install EGL libs (libegl1)."
    fi
fi

# 5. Optional: prefetch into HF cache (offline clusters; surfaces auth/network
#    failures here instead of mid-run). PREFETCH=all enables both.
if [ "${PREFETCH:-}" = "all" ]; then
    PREFETCH_MODELS=1
    PREFETCH_DATA=1
fi

if [ "${PREFETCH_MODELS:-0}" = "1" ]; then
    echo "==> Prefetching base (smolvla_libero) + its VLM (SmolVLM2-500M) into HF cache"
    python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("HuggingFaceVLA/smolvla_libero")
snapshot_download("HuggingFaceTB/SmolVLM2-500M-Instruct")
PY
fi

if [ "${PREFETCH_DATA:-0}" = "1" ]; then
    echo "==> Prefetching lerobot/libero dataset (~2GB) into HF cache"
    python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("lerobot/libero", repo_type="dataset", revision="v3.0")
PY
fi

echo
echo "Setup complete."
echo "Activate the venv with:   source $VENV_DIR/bin/activate"
echo "Frozen-base LIBERO-Pro smoke eval:"
echo "  MUJOCO_GL=egl python eval_libero_pro.py \\"
echo "    --policy.path=HuggingFaceVLA/smolvla_libero \\"
echo "    --env.type=libero --env.task=libero_object_swap --env.task_ids='[0]' \\"
echo "    --env.episode_length=30 --eval.n_episodes=1 --eval.batch_size=1 \\"
echo "    --policy.device=cuda --policy.use_amp=false \\"
echo "    --output_dir=outputs/eval/smoke_libero_pro"
