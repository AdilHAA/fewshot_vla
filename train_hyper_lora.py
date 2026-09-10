"""Train the hypernetwork online via stock `lerobot-train`.

The LIBERO-finetuned base is frozen end-to-end; only the hypernet trains, via
the flow-matching loss on all `lerobot/libero` tasks. Generalization is measured
on LIBERO-Pro, not here.

    python train_hyper_lora.py \
        --policy.type=hyper_lora_smolvla \
        --dataset.repo_id=lerobot/libero \
        --dataset.use_imagenet_stats=false \
        --policy.push_to_hub=false \
        --steps=100000 --batch_size=8 \
        --output_dir=outputs/hyper_lora_run

`smolvla_libero` was trained with a non-default narrow action expert
(expert_width_multiplier=0.5). We derive those SmolVLA fields from the base
checkpoint's config.json and inject them so the skeleton matches the base
state_dict. Explicit CLI flags win; resuming a checkpoint skips injection.
"""

import os
import sys

import src.hyper_lora  # noqa: F401 — registers the hyper_lora_smolvla policy type
import src.hyper_lora_traj  # noqa: F401 — registers the traj_hyper_lora_smolvla policy type
from src.hyper_lora import HyperLoRASmolVLAConfig
from src.hyper_lora.base_config import base_config_overrides
from lerobot.scripts.lerobot_train import main


class _TensorBoardLogger:
    """Drop-in for lerobot's WandBLogger writing scalars to TensorBoard instead.
    Enabled with env TENSORBOARD=1 (train.sh TB=1); needs no network or login.
    View with: tensorboard --logdir <output_dir>/tensorboard"""

    def __init__(self, cfg):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=os.path.join(str(cfg.output_dir), "tensorboard"))

    def log_dict(self, d, step, mode="train"):
        for k, v in d.items():
            if isinstance(v, (int, float)):
                self._writer.add_scalar(f"{mode}/{k}", v, step)
        self._writer.flush()

    def __getattr__(self, name):  # log_policy/log_video/...: ignore quietly
        return lambda *a, **k: None


def _inject_base_config_overrides() -> None:
    argv = sys.argv[1:]
    # Both policy types load the same frozen base, so both need the narrow-expert
    # SmolVLA config overrides derived from the base checkpoint.
    if not any(
        f"--policy.type={t}" in argv
        for t in ("hyper_lora_smolvla", "traj_hyper_lora_smolvla")
    ):
        return
    if any(a.startswith("--policy.path=") for a in argv):
        return  # resuming our own checkpoint: its config.json is authoritative

    base = HyperLoRASmolVLAConfig.base_smolvla_path
    for a in argv:
        if a.startswith("--policy.base_smolvla_path="):
            base = a.split("=", 1)[1]
    if not base or str(base).lower() == "none":
        return

    extra = base_config_overrides(str(base), argv)
    if extra:
        print(f"[train_hyper_lora] matching SmolVLA config to base {base!r}: "
              f"{len(extra)} override(s)")
        sys.argv += extra


def _patch_deterministic_episode_filters() -> None:
    """lerobot 0.5.1 feeds `--dataset.episodes` into Dataset.from_parquet as a
    pyarrow Expression. On some pyarrow/datasets combos the Expression does not
    pickle deterministically, so the `datasets` cache fingerprint changes on
    EVERY launch and the multi-GB arrow cache is rebuilt from scratch each time.
    Under accelerate DDP that is fatal, not just slow: ranks 1..N wait for rank 0
    at a 10-minute barrier while it rebuilds, then die with c10d
    'wait timeout after 600000ms' (observed on the LIBERO-90 finetune).
    The SAME filter as a plain DNF list of tuples hashes deterministically and
    pyarrow accepts it unchanged — verified: identical rows, one cache dir,
    second process reuses it without a rebuild."""
    import datasets as hf_datasets
    import lerobot.datasets.dataset_reader as _dr

    def load_nested_dataset(pq_dir, features=None, episodes=None):
        paths = sorted(pq_dir.glob("*/*.parquet"))
        if len(paths) == 0:
            raise FileNotFoundError(f"Provided directory does not contain any parquet file: {pq_dir}")
        filters = ([("episode_index", "in", sorted(int(e) for e in episodes))]
                   if episodes is not None else None)
        # Serialize dataset loads across the LOCAL node with a /tmp flock. With the
        # datasets cache on NFS, several DDP ranks entering the cache at once
        # deadlocked in its file locks (ranks frozen: 0 CPU, 0 IO, 0 majflt) while
        # rank 0 — which always loads ALONE before the barrier — never hung. /tmp
        # is node-local, so this flock always works; ranks simply take turns.
        import fcntl

        with open(f"/tmp/lerobot_dsload_{os.getuid()}.lock", "w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            try:
                return hf_datasets.Dataset.from_parquet([str(p) for p in paths],
                                                        filters=filters, features=features)
            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)

    # dataset_reader binds the name at import (`from ...io_utils import ...`),
    # so the override must land on the READER module, not on io_utils.
    _dr.load_nested_dataset = load_nested_dataset


def _patch_ddp_timeout() -> None:
    """Raise the process-group timeout for DDP runs (env DDP_TIMEOUT_S, default
    3600s; torch's NCCL default is 600s). lerobot's launch order is: rank 0
    builds the dataset (a cold arrow-cache build over NFS runs ~20 min), the
    other ranks wait at a barrier — with the 10-minute default they die with
    c10d 'wait timeout after 600000ms' before rank 0 arrives. lerobot creates
    the Accelerator itself with no timeout knob, so the kwargs handler is
    injected here. Append-only: an InitProcessGroupKwargs already passed by the
    caller wins."""
    import datetime

    import accelerate
    from accelerate.utils import InitProcessGroupKwargs

    seconds = int(os.environ.get("DDP_TIMEOUT_S", "3600"))
    orig = accelerate.Accelerator.__init__

    def patched(self, *args, **kwargs):
        handlers = list(kwargs.get("kwargs_handlers") or [])
        if not any(isinstance(h, InitProcessGroupKwargs) for h in handlers):
            handlers.append(InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=seconds)))
        kwargs["kwargs_handlers"] = handlers
        orig(self, *args, **kwargs)

    accelerate.Accelerator.__init__ = patched


def _patch_datasets_soft_filelock() -> None:
    """DATASETS_SOFT_LOCK=1: make `datasets` take its cache lock with SoftFileLock
    (lock = atomic creation of a file) instead of fcntl. On this node the cache
    lives on an NFS4 mount with local_lock=none, i.e. fcntl locks go to the NFS
    server — the first process (rank 0) acquired it fine, every later rank blocked
    in the kernel forever (0 CPU, 0 I/O, 0 page faults) and the DDP run hung at
    rank 0's first collective. SoftFileLock needs no server-side lock state.
    Caveat: a process killed while holding it leaves <cache>/**/*.lock behind —
    delete it if a later run stalls at 'Creating dataset'."""
    if os.environ.get("DATASETS_SOFT_LOCK") != "1":
        return
    import filelock
    import datasets.builder as _b
    import datasets.utils._filelock as _fl

    _b.FileLock = filelock.SoftFileLock          # download_and_prepare()'s lock
    _fl.FileLock = filelock.SoftFileLock
    print("[train] DATASETS_SOFT_LOCK: datasets cache locks -> SoftFileLock")


def _enable_stack_dump_on_usr1() -> None:
    """`kill -USR1 <pid>` makes the process print every thread's Python stack to
    stderr (faulthandler; no ptrace needed — py-spy is forbidden on the cluster).
    The DDP hangs on the 8xA100 node could not be located otherwise: ranks sat in
    R/D states with no I/O and no page faults for an hour."""
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


def _patch_fast_index_mapping() -> None:
    """lerobot 0.5.1 builds the episode-filtered index map as
    `enumerate(self.hf_dataset["index"])`. With datasets>=3 that is a lazy Column
    iterated ROW BY ROW through the dataset's set_transform (hf_transform_to_torch),
    i.e. every PNG of every frame is decoded and to_tensor'ed just to read one
    int: ~20 min of CPU per process on LIBERO-90 ('Creating dataset' on a WARM
    cache), and under DDP the three non-main ranks spend it while rank 0 spins at
    the next barrier — the 'hang' at 'Loading weights from local directory'
    (found via the SIGUSR1 stack dumps). Read the column from arrow instead:
    identical mapping (verified), milliseconds. Falls back to the original when
    the dataset carries an indices mapping (select/shuffle), which from_parquet
    never produces here."""
    import lerobot.datasets.dataset_reader as _dr

    orig = _dr.DatasetReader._build_index_mapping

    def fast(self):
        self._absolute_to_relative_idx = None
        if self.episodes is None or self.hf_dataset is None:
            return
        if getattr(self.hf_dataset, "_indices", None) is not None:
            return orig(self)
        col = self.hf_dataset.data.column("index").to_pylist()
        self._absolute_to_relative_idx = {int(a): r for r, a in enumerate(col)}

    _dr.DatasetReader._build_index_mapping = fast


if __name__ == "__main__":
    _enable_stack_dump_on_usr1()
    _inject_base_config_overrides()
    _patch_deterministic_episode_filters()
    _patch_fast_index_mapping()
    _patch_ddp_timeout()
    _patch_datasets_soft_filelock()
    if os.environ.get("TENSORBOARD") == "1":
        # The train loop instantiates whatever `WandBLogger` names in its module
        # namespace; rebinding it routes all metric logging to TensorBoard without
        # touching lerobot (wandb is never imported).
        import lerobot.scripts.lerobot_train as _lt

        _lt.WandBLogger = _TensorBoardLogger
    main()
