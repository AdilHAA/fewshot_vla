"""Render sim-recolor variants of LIBERO demo opening clips by MuJoCo state replay.

Recolors the scene objects via the libero_pro variant machinery, replays the recorded
per-step sim states of each demo (no physics stepping), and captures the first T
agentview frames — the exact same motion with a different appearance. Demo states come
from the original LIBERO HDF5 datasets; hdf5 demos are matched to lerobot episodes by
comparing action prefixes. Output feeds `build_xpair_cache.py --rendered_dir`.

`--cam_jitters N` additionally renders N unrecolored variants per episode from a
perturbed agentview camera (same states, shifted viewpoint) — saved as `ep*_camJ.npz`
and picked up by the cache builder like any other rendered variant.

  python scripts/render_recolor_clips.py --colors red,green --num_frames 16 \
      --cam_jitters 2 --out outputs/rendered_recolor
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.traj_data.augment import opening_clip_indices

logger = logging.getLogger("render_recolor")

BASE_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")

# BDDL parsing (local copies so this module imports without LIBERO installed).
_OBJ_BLOCK = re.compile(r"\(:(?:objects|fixtures)(.*?)\)", re.DOTALL)
_TYPE_LINE = re.compile(r"^(\s*\S+\s+-\s+)(\S+)(\s*)$", re.MULTILINE)
_ARENA_SURFACES = {"floor", "table", "kitchen_table", "living_room_table", "study_table"}


def bddl_categories(text: str) -> set[str]:
    """Object/fixture categories declared in one BDDL (arena surfaces excluded)."""
    cats: set[str] = set()
    for block in _OBJ_BLOCK.finditer(text):
        for m in _TYPE_LINE.finditer(block.group(1)):
            cats.add(m.group(2))
    return cats - _ARENA_SURFACES


def recolor_bddl(text: str, color: str, resolvable) -> tuple[str, list[str]]:
    """Rewrite category (type) tokens in :objects/:fixtures to `{color}_` variants.

    Only the type token changes — instance names and predicates stay, so the model's
    body/joint layout and the recorded state vector remain valid. Categories whose
    recolored key `resolvable()` rejects are left as-is. Returns (text, new_keys)."""
    keys: list[str] = []

    def _line(lm):
        cat = lm.group(2)
        new = f"{color}_{cat}"
        if cat in _ARENA_SURFACES or cat.startswith(f"{color}_") or not resolvable(new):
            return lm.group(0)
        if new not in keys:
            keys.append(new)
        return f"{lm.group(1)}{new}{lm.group(3)}"

    def _block(bm):
        return bm.group(0).replace(bm.group(1), _TYPE_LINE.sub(_line, bm.group(1)), 1)

    return _OBJ_BLOCK.sub(_block, text), keys


def match_demo(ep_actions: np.ndarray, demo_actions: dict, atol: float = 1e-4):
    """Return the hdf5 demo name whose action prefix matches the lerobot episode's."""
    n = ep_actions.shape[0]
    for name, acts in demo_actions.items():
        if acts.shape[0] >= n and np.allclose(acts[:n], ep_actions, atol=atol):
            return name
    return None


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def jitter_camera(base_pos: np.ndarray, base_quat: np.ndarray, rng,
                  dpos: float, drot_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Perturbed (pos, quat) for a fixed camera: uniform position offset plus a small
    axis-angle rotation composed onto the base orientation (MuJoCo wxyz quats)."""
    pos = base_pos + rng.uniform(-dpos, dpos, size=3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    half = np.deg2rad(rng.uniform(-drot_deg, drot_deg)) / 2.0
    w1, (x1, y1, z1) = np.cos(half), np.sin(half) * axis
    w2, x2, y2, z2 = base_quat
    quat = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return pos, quat / np.linalg.norm(quat)


# --- sim side (lazy imports; needs LIBERO + EGL) ---------------------------------------

def _libero_task_map() -> dict:
    """normalized instruction -> (suite, Task) over the 4 base suites."""
    from libero.libero import benchmark as lb

    out = {}
    for suite in BASE_SUITES:
        bm = lb.get_benchmark_dict()[suite]()
        for tid in range(bm.get_num_tasks()):
            task = bm.get_task(tid)
            out[_norm(task.language)] = (suite, task)
    return out


def _make_env(bddl_path: Path):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path), camera_heights=256, camera_widths=256)
    env.reset()
    return env


def _render_states(env, states: np.ndarray, flip: bool) -> np.ndarray:
    frames = []
    for s in states:
        env.sim.set_state_from_flattened(s.astype(np.float64))
        env.sim.forward()
        img = env.sim.render(camera_name="agentview", height=256, width=256)
        frames.append(img[::-1] if flip else img)
    return np.stack(frames).astype(np.uint8)


def _lerobot_episodes(repo_id: str, revision: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.data.libero import prefetch_all_data_parquets

    prefetch_all_data_parquets(repo_id, revision)
    ds = LeRobotDataset(repo_id, revision=revision)
    eps = ds.meta.episodes

    def _erow(i):
        return eps.iloc[i].to_dict() if hasattr(eps, "iloc") else eps[i]

    rows, acc = [], 0
    for ep in range(ds.meta.total_episodes):
        row = _erow(ep)
        if row.get("dataset_from_index") is not None:
            start = int(row["dataset_from_index"])
            length = int(row["dataset_to_index"]) - start
        else:
            start, length = acc, int(row["length"])
            acc += length
        rows.append((ep, start, length))
    return ds, rows


def _detect_flip(env, state0: np.ndarray, ref_hwc: np.ndarray) -> tuple[bool, float]:
    """Render the unrecolored state and pick the orientation matching the dataset."""
    env.sim.set_state_from_flattened(state0.astype(np.float64))
    env.sim.forward()
    img = env.sim.render(camera_name="agentview", height=256, width=256)
    img = img.astype(np.float32) / 255.0
    mae_up = float(np.abs(img - ref_hwc).mean())
    mae_fl = float(np.abs(img[::-1] - ref_hwc).mean())
    return mae_fl < mae_up, min(mae_up, mae_fl)


def _contact_sheet(out_dir: Path, max_tiles: int = 12) -> None:
    import imageio.v2 as imageio

    tiles = []
    for path in sorted(out_dir.glob("ep*.npz"))[:max_tiles]:
        tiles.append(np.load(path)["frames"][0])
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    h, w, c = tiles[0].shape
    sheet = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, cc = divmod(i, cols)
        sheet[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = t
    imageio.imwrite(out_dir / "contact_sheet.png", sheet)


def main(argv=None):  # pragma: no cover (GPU/EGL + datasets)
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="lerobot/libero")
    p.add_argument("--revision", default="v3.0")
    p.add_argument("--hdf5_root", default=None,
                   help="LIBERO hdf5 datasets root (default: get_libero_path('datasets'))")
    p.add_argument("--out", default="outputs/rendered_recolor")
    p.add_argument("--colors", default="red,green")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--gate_mae", type=float, default=0.10,
                   help="max MAE between an unrecolored render and the dataset frame")
    p.add_argument("--cam_jitters", type=int, default=0,
                   help="extra unrecolored variants per episode from a jittered camera")
    p.add_argument("--cam_pos_jitter", type=float, default=0.05,
                   help="camera position jitter, meters (uniform per axis)")
    p.add_argument("--cam_rot_jitter", type=float, default=5.0,
                   help="camera rotation jitter, degrees (axis-angle)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    import h5py

    from libero.libero import get_libero_path
    from src.libero_pro.objects import _COLORS, _resolve, register_object_keys

    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    bad = [c for c in colors if c not in _COLORS]
    if bad:
        raise SystemExit(f"unknown colors {bad}; available: {sorted(_COLORS)}")
    hdf5_root = Path(args.hdf5_root or get_libero_path("datasets"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bddl_dir = out_dir / "_bddl"
    bddl_dir.mkdir(exist_ok=True)

    ds, rows = _lerobot_episodes(args.repo_id, args.revision)
    img_key = ds.meta.camera_keys[0]
    task_map = _libero_task_map()

    # Group lerobot episodes by task.
    by_task: dict[int, list] = {}
    for ep, start, length in rows:
        first = ds[start]
        by_task.setdefault(int(first["task_index"]), []).append((ep, start, length, first))

    need = args.num_frames * args.stride
    done = skipped = 0
    for ti, (task_index, ep_list) in enumerate(sorted(by_task.items())):
        if ti % args.num_shards != args.shard:
            continue
        task_str = ep_list[0][3].get("task", "")
        hit = task_map.get(_norm(task_str))
        if hit is None:
            logger.warning("task %d %r: no LIBERO task matches — skipped", task_index, task_str)
            skipped += len(ep_list)
            continue
        suite, task = hit
        h5_path = hdf5_root / task.problem_folder / f"{task.name}_demo.hdf5"
        if not h5_path.exists():
            logger.warning("task %d: missing %s (download the LIBERO datasets) — skipped",
                           task_index, h5_path)
            skipped += len(ep_list)
            continue
        src_bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        text = src_bddl.read_text()

        with h5py.File(h5_path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda n: int(n.split("_")[-1]))
            demo_actions = {d: f[f"data/{d}/actions"][:8].astype(np.float32) for d in demos}
            demo_states = {d: f[f"data/{d}/states"][:need].astype(np.float64) for d in demos}

        # Match every episode to its hdf5 demo by action prefix.
        matches = []
        for ep, start, length, first in ep_list:
            ep_act = np.stack(
                [np.asarray(ds[start + i]["action"], dtype=np.float32) for i in range(5)])
            demo = match_demo(ep_act, demo_actions)
            if demo is None:
                logger.warning("episode %d (task %d): no action-matching demo — skipped",
                               ep, task_index)
                skipped += 1
            else:
                matches.append((ep, start, demo))
        if not matches:
            continue

        # Orientation + camera + mapping gate on the unrecolored scene.
        env = _make_env(src_bddl)
        ep0, start0, demo0 = matches[0]
        ref = np.asarray(ds[start0][img_key], dtype=np.float32)          # (C,H,W) 0-1
        ref = np.transpose(ref, (1, 2, 0))
        flip, mae = _detect_flip(env, demo_states[demo0][0], ref)
        env.close()
        logger.info("task %d [%s/%s]: gate mae=%.4f flip=%s (%d episodes)",
                    task_index, suite, task.name, mae, flip, len(matches))
        if mae > args.gate_mae:
            logger.warning("task %d: gate MAE %.4f > %.2f — render does not match the "
                           "dataset (camera/orientation/mapping); skipped", task_index,
                           mae, args.gate_mae)
            skipped += len(matches)
            continue

        for color in colors:
            mod_text, keys = recolor_bddl(text, color, lambda k: _resolve(k)[0] is not None)
            if not keys:
                logger.warning("task %d: no recolorable categories for %s — skipped",
                               task_index, color)
                continue
            register_object_keys(keys)
            mod_bddl = bddl_dir / f"{task.name}_{color}.bddl"
            mod_bddl.write_text(mod_text)
            env = _make_env(mod_bddl)
            for ep, start, demo in matches:
                dst = out_dir / f"ep{ep:05d}_{color}.npz"
                if dst.exists():
                    continue
                states = demo_states[demo]
                idx = opening_clip_indices(states.shape[0], args.num_frames, args.stride)
                try:
                    frames = _render_states(env, states[idx], flip)
                except Exception as e:  # state-dim mismatch etc. — skip, keep going
                    logger.warning("episode %d/%s: render failed: %s", ep, color, e)
                    skipped += 1
                    continue
                np.savez_compressed(dst, frames=frames, episode=ep, color=color,
                                    task_index=task_index)
                done += 1
            env.close()

        if args.cam_jitters:
            env = _make_env(src_bddl)
            cam_id = env.sim.model.camera_name2id("agentview")
            base_pos = env.sim.model.cam_pos[cam_id].copy()
            base_quat = env.sim.model.cam_quat[cam_id].copy()
            for ep, start, demo in matches:
                states = demo_states[demo]
                idx = opening_clip_indices(states.shape[0], args.num_frames, args.stride)
                for j in range(args.cam_jitters):
                    dst = out_dir / f"ep{ep:05d}_cam{j}.npz"
                    if dst.exists():
                        continue
                    rng = np.random.default_rng(ep * 7919 + j)
                    pos, quat = jitter_camera(base_pos, base_quat, rng,
                                              args.cam_pos_jitter, args.cam_rot_jitter)
                    env.sim.model.cam_pos[cam_id] = pos
                    env.sim.model.cam_quat[cam_id] = quat
                    try:
                        frames = _render_states(env, states[idx], flip)
                    except Exception as e:
                        logger.warning("episode %d/cam%d: render failed: %s", ep, j, e)
                        skipped += 1
                        continue
                    np.savez_compressed(dst, frames=frames, episode=ep, color=f"cam{j}",
                                        task_index=task_index)
                    done += 1
            env.close()

    _contact_sheet(out_dir)
    print(f"rendered {done} clips to {out_dir} ({skipped} skipped); "
          f"inspect {out_dir}/contact_sheet.png")


if __name__ == "__main__":  # pragma: no cover
    main()
