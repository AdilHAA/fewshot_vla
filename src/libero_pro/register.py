"""Install LIBERO-Pro task files and register them as LIBERO benchmarks.

Registering a perturbed suite = (1) copy its BDDL/init into the LIBERO paths,
(2) add a ``task_maps`` entry, (3) register a ``Benchmark`` subclass whose
``__name__.lower()`` is the suite name lerobot requests. The success predicate
is built from the BDDL, so the Task axis comes along for free.

The instruction is parsed from each BDDL's ``(:language ...)`` field, not the
filename: across axes the filenames are identical and the Semantic (``_lan``)
perturbation lives inside the BDDL.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from libero.libero import benchmark as _lb
from libero.libero import get_libero_path

from .fetch import fetch_libero_pro
from .objects import object_categories_in_bddls, register_object_keys

logger = logging.getLogger(__name__)

# suffix in zhouxueyang/LIBERO-Pro → generalization axis
PERTURBATIONS: dict[str, str] = {
    "lan": "semantic",
    "object": "object",
    "swap": "position",
    "task": "task",
}
BASE_SUITES: tuple[str, ...] = (
    "libero_10",
    "libero_goal",
    "libero_object",
    "libero_spatial",
)

_LANG_RE = re.compile(r"\(:language\s+(.*?)\)", re.DOTALL)
_REGISTERED: set[str] = set()


def _parse_language(bddl_path: Path) -> str:
    """Extract the instruction from a BDDL ``(:language ...)`` field."""
    m = _LANG_RE.search(bddl_path.read_text())
    if not m:
        raise ValueError(f"No (:language ...) field in {bddl_path}")
    return m.group(1).strip().strip('"').strip()


def _install_suite(snapshot_root: Path, suite: str) -> tuple[Path, Path]:
    """Copy one perturbed suite's BDDL+init into the LIBERO paths (idempotent).

    Returns (dst_bddl_dir, dst_init_dir).
    """
    src_bddl = snapshot_root / "bddl_files" / suite
    src_init = snapshot_root / "init_files" / suite
    dst_bddl = Path(get_libero_path("bddl_files")) / suite
    dst_init = Path(get_libero_path("init_states")) / suite
    dst_bddl.mkdir(parents=True, exist_ok=True)
    dst_init.mkdir(parents=True, exist_ok=True)

    def _sync(src_dir: Path, dst_dir: Path, pattern: str) -> None:
        for src in sorted(src_dir.glob(pattern)):
            dst = dst_dir / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)

    _sync(src_bddl, dst_bddl, "*.bddl")
    _sync(src_init, dst_init, "*.pruned_init")
    return dst_bddl, dst_init


def _build_task_map(suite: str, bddl_dir: Path, init_dir: Path) -> dict:
    """Build the ``{task_name: Task}`` map for a perturbed suite."""
    task_map: dict[str, _lb.Task] = {}
    for bddl in sorted(bddl_dir.glob("*.bddl")):
        name = bddl.stem
        init_file = init_dir / f"{name}.pruned_init"
        if not init_file.exists():
            raise FileNotFoundError(
                f"Missing init for {suite}/{name}: expected {init_file}"
            )
        task_map[name] = _lb.Task(
            name=name,
            language=_parse_language(bddl),
            problem="Libero",
            problem_folder=suite,
            bddl_file=f"{name}.bddl",
            init_states_file=f"{name}.pruned_init",
        )
    return task_map


def _register_suite(suite: str, task_map: dict) -> None:
    """Inject the task map and register a Benchmark subclass for ``suite``."""
    n = len(task_map)
    if n != 10:
        # _make_benchmark indexes via task_orders (perms of 0..9); fail loud
        # rather than IndexError deep inside lerobot.
        raise ValueError(f"Suite {suite}: expected 10 tasks, got {n}")

    _lb.task_maps[suite] = task_map

    def __init__(self, task_order_index: int = 0) -> None:
        _lb.Benchmark.__init__(self, task_order_index=task_order_index)
        self.name = suite
        self._make_benchmark()

    cls = type(suite, (_lb.Benchmark,), {"__init__": __init__})
    _lb.register_benchmark(cls)  # keyed by cls.__name__.lower() == suite


def perturbed_suite_names() -> list[str]:
    """Registered LIBERO-Pro suite names (``<base>_<suffix>``)."""
    return sorted(_REGISTERED)


def ensure_registered(repo_id: str = "zhouxueyang/LIBERO-Pro") -> list[str]:
    """Fetch → install → register all LIBERO-Pro suites. Idempotent.

    Suites are discovered from the snapshot (robust to the dataset gaining or
    dropping axes), not hard-coded.
    """
    if _REGISTERED:
        return perturbed_suite_names()

    root = fetch_libero_pro(repo_id)
    suite_dirs = sorted(p.name for p in (root / "bddl_files").iterdir() if p.is_dir())
    if not suite_dirs:
        raise RuntimeError(f"No bddl_files/<suite> dirs in snapshot {root}")

    skipped: list[str] = []
    for suite in suite_dirs:
        if suite in _lb.get_benchmark_dict():
            _REGISTERED.add(suite)
            continue
        # A suite that doesn't lay out as exactly 10 <name>.bddl + <name>.pruned_init
        # (e.g. a newly-added axis with a different naming/nesting scheme) must not
        # abort registration of the suites we do use — skip it with a warning.
        try:
            bddl_dir, init_dir = _install_suite(root, suite)
            if suite.endswith("_object"):
                # Object axis: synthesize the visually-changed objects the BDDLs
                # rename to but ship no assets for, else env build KeyErrors.
                register_object_keys(object_categories_in_bddls(bddl_dir))
            task_map = _build_task_map(suite, bddl_dir, init_dir)
            _register_suite(suite, task_map)
            _REGISTERED.add(suite)
        except Exception as e:
            skipped.append(f"{suite} ({type(e).__name__}: {e})")

    if not _REGISTERED:
        raise RuntimeError(
            f"No LIBERO-Pro suite registered from {root}; all {len(suite_dirs)} "
            f"skipped: {'; '.join(skipped)}"
        )
    logger.info("Registered %d LIBERO-Pro suites: %s",
                len(_REGISTERED), ", ".join(perturbed_suite_names()))
    if skipped:
        logger.warning("Skipped %d LIBERO-Pro suite dir(s): %s",
                       len(skipped), ", ".join(skipped))
    return perturbed_suite_names()
