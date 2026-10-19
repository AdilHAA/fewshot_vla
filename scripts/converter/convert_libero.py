#!/usr/bin/env python3
"""Filter LIBERO data and retime retained steps to 10 FPS.

Input: aligned pre-action RGB observations, raw relative OSC actions (7D),
absolute EEF axis-angle + finger qpos state (8D). HDF5 keeps the source gripper
encoding; LeRobot mode requires -1=open/+1=close.
No physical resampling, action aggregation, image rotation or simulator replay.
HDF5 mode preserves HDF5 and needs only numpy/h5py (Python >=3.10).
LeRobot mode needs lerobot==0.5.1 av==15.1.0 (Python >=3.12).
This file has no imports from the project.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np

OPENVLA_SOURCE = "https://github.com/moojink/openvla-oft/blob/main/experiments/robot/libero/regenerate_libero_dataset.py"
DEFAULT_COLUMNS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
TIME_NAMES = {"time", "timestamp", "timestamps"}
INDEX_NAMES = {"source_indices", "source_frame_indices", "source_step_indices", "original_indices",
               "original_frame_indices"}


def hdf5_metadata(trace):
    import h5py

    containers = [("attrs", "/", dict(trace.attrs))]
    if "metadata" in trace.attrs:
        containers.append(("json_attr", "/", json.loads(trace.attrs["metadata"])))
    if "metadata" in trace:
        node = trace["metadata"]
        if isinstance(node, h5py.Group):
            containers.append(("attrs", "metadata", dict(node.attrs)))
            for key, value in node.items():
                if isinstance(value, h5py.Dataset) and value.shape == ():
                    containers.append(("scalar", f"metadata/{key}", {key: value[()]}))
        elif node.shape == ():
            containers.append(("json_dataset", "metadata", json.loads(node[()])))
        else:
            raise ValueError("metadata must be a group or a scalar JSON dataset")
    for key in ("fps", "action_offset", "num_frames", "num_steps", "length"):
        if key in trace and isinstance(trace[key], h5py.Dataset) and trace[key].shape == ():
            containers.append(("scalar", key, {key: trace[key][()]}))
    if any(not isinstance(values, dict) for _, _, values in containers):
        raise ValueError("metadata JSON must contain an object")
    return containers


def metadata_value(containers, key, fallback=None):
    values = [float(values[key]) for _, _, values in containers if key in values]
    if fallback is not None:
        values.append(float(fallback))
    if not values or not np.isfinite(values).all() or any(v != values[0] for v in values):
        raise ValueError(f"missing or conflicting {key}; supply --{key.replace('_', '-')} if absent")
    return values[0]


def inspect_hdf5(trace, args):
    import h5py

    containers = hdf5_metadata(trace)
    fps = metadata_value(containers, "fps", args.fps)
    offset = metadata_value(containers, "action_offset", args.action_offset)
    if fps not in (10, 20) or offset != 0:
        raise ValueError("requires 10/20 FPS and pre-action alignment action_offset=0")
    if "_conversion" in trace or "stats" in trace:
        raise ValueError("input has _conversion/stats; resolve existing processing/statistics explicitly")
    actions, states = trace[args.actions_key], trace[args.state_key]
    if actions.ndim != 2 or actions.shape[1] != 7 or states.shape != (len(actions), 8):
        raise ValueError("expected actions[T,7] and observation_state[T,8]")
    count = len(actions)
    if (not np.isfinite(states[:]).all() or not np.isfinite(actions[:]).all()
            or np.any(np.abs(actions[:]) > 1 + 1e-6)):
        raise ValueError("requires finite state and raw relative actions in [-1,1]")
    datasets = {}

    def remember_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj

    trace.visititems(remember_dataset)
    cameras = args.camera_keys or [key for key, value in datasets.items()
                                   if value.shape == (count, 256, 256, 3) and value.dtype == np.uint8]
    if (len(cameras) != 2 or len(set(cameras)) != 2
            or any(datasets[key].shape != (count, 256, 256, 3) or datasets[key].dtype != np.uint8
                   for key in cameras)):
        raise ValueError("requires exactly two uint8 RGB cameras [T,256,256,3]; use --camera-keys")
    timed = {args.actions_key, args.state_key, *cameras, *args.time_keys}
    static = set(args.static_keys)
    if timed & static or (timed | static) - datasets.keys():
        raise ValueError("invalid or overlapping --time-keys/--static-keys")
    for key, data in datasets.items():
        name = key.rsplit("/", 1)[-1]
        if key in static or key == "metadata" or key.startswith("metadata/") or data.ndim == 0:
            continue
        if name in TIME_NAMES | INDEX_NAMES | {"frame_index", "rewards", "dones", "success"}:
            timed.add(key)
        if key not in timed and len(data) in (count, count + 1):
            raise ValueError(f"ambiguous dataset {key}: classify with --time-keys or --static-keys")
    for key in timed:
        if datasets[key].ndim == 0 or len(datasets[key]) != count:
            raise ValueError(f"misaligned temporal dataset: {key}")
        name = key.rsplit("/", 1)[-1]
        if name in TIME_NAMES:
            times = datasets[key][:]
            if (not np.issubdtype(times.dtype, np.floating)
                    or times.shape != (count,) or not np.isfinite(times).all() or not np.allclose(
                    np.diff(times), 1 / fps, rtol=0, atol=1e-4)):
                raise ValueError(f"invalid source timestamps: {key}")
        if name == "frame_index" and not np.array_equal(datasets[key][:], np.arange(count)):
            raise ValueError(f"invalid frame indices: {key}")
    owners = {key for key in timed}
    owners.update(parent.as_posix() for key in timed for parent in Path(key).parents
                  if parent.as_posix() not in (".", "metadata"))
    for owner in sorted(owners):
        if trace[owner].attrs:
            containers.append(("attrs", owner, dict(trace[owner].attrs)))
    metadata_value(containers, "fps", fps)
    metadata_value(containers, "action_offset", offset)
    for _, _, values in containers:
        for key in ("num_frames", "num_steps"):
            if key in values and values[key] != count:
                raise ValueError(f"{key} disagrees with action count: {values[key]} != {count}")
    keep = openvla_keep_indices(actions[:])
    if len(keep) == 0:
        raise ValueError("all actions are no-ops; refusing to silently drop an episode/task")
    return keep, timed, containers, fps


def write_hdf5_metadata(trace, containers, count, keep):
    import h5py

    for kind, path, original in containers:
        updated = dict(original)
        for key in ("fps", "action_offset", "num_frames", "num_steps", "length"):
            if key in updated:
                if key == "fps":
                    updated[key] = 10
                elif key == "action_offset":
                    updated[key] = 0
                elif updated[key] == count:
                    updated[key] = len(keep)
        for key in TIME_NAMES | INDEX_NAMES | {"frame_index"}:
            if isinstance(updated.get(key), (list, np.ndarray)) and len(updated[key]) == count:
                selected = np.asarray(updated[key])[keep]
                updated[key] = (np.arange(len(keep)) / 10 if key in TIME_NAMES else
                                np.arange(len(keep)) if key == "frame_index" else selected)
                if kind in ("json_attr", "json_dataset"):
                    updated[key] = updated[key].tolist()
        if kind in ("json_attr", "json_dataset"):
            updated.update(fps=10, action_offset=0)
            encoded = json.dumps(updated)
            if kind == "json_attr":
                trace[path].attrs["metadata"] = encoded
            else:
                attrs = dict(trace[path].attrs)
                del trace[path]
                trace.create_dataset(path, data=encoded, dtype=h5py.string_dtype("utf-8")).attrs.update(attrs)
        elif kind == "attrs":
            for key in {"fps", "action_offset", "num_frames", "num_steps", "length", "frame_index"} | TIME_NAMES | INDEX_NAMES:
                if key in updated:
                    trace[path].attrs[key] = updated[key]
        else:
            trace[path][()] = updated[path.rsplit("/", 1)[-1]]
    trace.attrs.update(fps=10, action_offset=0)


def write_hdf5_episode(source_path, destination, relative_path, args):
    import h5py

    with h5py.File(source_path, "r") as source:
        keep, timed, containers, fps = inspect_hdf5(source, args)
        count = len(source[args.actions_key])
        with h5py.File(destination, "w") as output:
            output.attrs.update(dict(source.attrs))

            def copy_node(name, node):
                if isinstance(node, h5py.Group):
                    output.require_group(name).attrs.update(dict(node.attrs))
                elif name not in timed:
                    source.copy(node, output, name=name)
                else:
                    options = {"compression": node.compression, "compression_opts": node.compression_opts,
                               "shuffle": node.shuffle, "fletcher32": node.fletcher32}
                    target = output.create_dataset(name, shape=(len(keep), *node.shape[1:]), dtype=node.dtype,
                                                   **options)
                    target.attrs.update(dict(node.attrs))
                    for start in range(0, len(keep), 32):
                        target[start:start + 32] = node[keep[start:start + 32]]
                    leaf = name.rsplit("/", 1)[-1]
                    if leaf in TIME_NAMES:
                        if not np.issubdtype(node.dtype, np.floating):
                            raise ValueError(f"timestamps must have floating dtype: {name}")
                        target[:] = np.arange(len(keep)) / 10
                    elif leaf == "frame_index":
                        target[:] = np.arange(len(keep))

            source.visititems(copy_node)
            write_hdf5_metadata(output, containers, count, keep)
            audit = output.create_group("_conversion")
            audit.create_group("source_attrs").attrs.update(dict(source.attrs))
            if "metadata" in source:
                source.copy("metadata", audit, name="source_metadata")
            audit.attrs.update(source_file=relative_path.as_posix(), source_fps=fps, output_fps=10,
                               source_length=count, output_length=len(keep), threshold=1e-4,
                               no_op_rule="openvla-oft", no_op_source=OPENVLA_SOURCE,
                               time_conversion="retime_retained_steps", simulator_replay=False)
            audit.create_dataset("input_row_indices", data=keep)
            audit.create_dataset("input_timestamps", data=keep / fps)
            for key in timed:
                if key.rsplit("/", 1)[-1] in TIME_NAMES:
                    audit.create_dataset("original_" + key.replace("/", "__"), data=source[key][keep])
        with h5py.File(destination, "r") as output:
            for key in timed:
                leaf = key.rsplit("/", 1)[-1]
                for start in range(0, len(keep), 32):
                    stop = min(start + 32, len(keep))
                    expected = (np.arange(start, stop) / 10 if leaf in TIME_NAMES else
                                np.arange(start, stop) if leaf == "frame_index" else source[key][keep[start:stop]])
                    if not np.array_equal(output[key][start:stop], expected.astype(output[key].dtype)):
                        raise ValueError(f"written dataset differs from selected source rows: {key}")
    return count, len(keep)


def convert_hdf5(args):
    import h5py

    source, destination = args.source.resolve(), args.out.resolve()
    pending = destination.with_name(destination.name + ".partial")
    if destination.exists() or pending.exists():
        raise FileExistsError("output or .partial directory exists; choose a new --out")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("source and output must be separate directories")
    base = source.parent if source.is_file() else source
    paths = [source] if source.is_file() else sorted(source.rglob(args.pattern))
    if not paths:
        raise ValueError("no HDF5 episodes matched --pattern")
    # Preflight every episode before writing so missing metadata/empty episodes cannot alter a split silently.
    for path in paths:
        with h5py.File(path, "r") as trace:
            inspect_hdf5(trace, args)
    pending.mkdir(parents=True)
    before, after = 0, 0
    for path in paths:
        relative = path.relative_to(base)
        target = pending / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        old, new = write_hdf5_episode(path, target, relative, args)
        before, after = before + old, after + new
        print(f"{relative}: {old} -> {new} frames", flush=True)
    pending.rename(destination)
    print(f"Saved {len(paths)} HDF5 episodes: {before} -> {after} frames; removed {before - after} no-ops")


def openvla_keep_indices(actions, threshold=1e-4):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        raise ValueError("expected finite relative actions with shape (T, 7)")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be finite and positive")
    keep, previous = [], None
    for i, action in enumerate(actions):
        # OpenVLA compares with the last retained action; initial stationary steps are dropped.
        if np.linalg.norm(action[:6]) < threshold and (previous is None or action[6] == previous):
            continue
        keep.append(i)
        previous = action[6]
    return np.asarray(keep, dtype=np.int64)


def install_safe_image_statistics():
    from lerobot.datasets import compute_stats

    class FloatImageStats(compute_stats.RunningQuantileStats):
        def update(self, batch):
            # LeRobot 0.5.1 otherwise squares uint8 images before casting, overflowing variance.
            return super().update(batch.astype(np.float64) if batch.dtype == np.uint8 else batch)

    compute_stats.RunningQuantileStats = FloatImageStats


def episode_data(source, episode_index):
    meta = source.meta.episodes[episode_index]
    start, stop = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    data = source.hf_dataset.with_format("numpy")[start:stop]
    count = stop - start
    if (count != meta["length"] or count < 1
            or not np.array_equal(data["index"], np.arange(start, stop))
            or not np.array_equal(data["frame_index"], np.arange(count))
            or not np.all(data["episode_index"] == episode_index)
            or not np.allclose(data["timestamp"], np.arange(count) / source.fps, atol=1e-4, rtol=0)):
        raise ValueError(f"episode {episode_index}: inconsistent indices/timestamps/length")
    actions, states = data["action"], data["observation.state"]
    if actions.shape != (count, 7) or states.shape != (count, 8) or not np.isfinite(states).all():
        raise ValueError(f"episode {episode_index}: invalid state/action dimensions or values")
    if (not np.isfinite(actions).all() or np.any(np.abs(actions[:, :6]) > 1 + 1e-6)
            or not np.isin(actions[:, 6], [-1, 1]).all()):
        raise ValueError(f"episode {episode_index}: requires raw relative OSC actions and -1/+1 gripper")
    if not np.isin(data["task_index"], source.meta.tasks["task_index"].to_numpy()).all():
        raise ValueError(f"episode {episode_index}: unknown task_index")
    return data


def selected_video_frames(path, timestamps, fps, shape):
    import av

    target_index, tolerance = 0, 1e-3
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if abs(float(stream.average_rate) - fps) > 1e-6:
            raise ValueError(f"video FPS disagrees with metadata: {path}")
        start = max(0, float(timestamps[0]) - 1 / fps)
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError(f"video has no frame timestamps: {path}")
            target = timestamps[target_index]
            if frame.time < target - tolerance:
                continue
            if abs(frame.time - target) > tolerance:
                raise ValueError(f"missing video frame at {target:.6f}s: {path}")
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != tuple(shape):
                raise ValueError(f"video shape disagrees with metadata: {path}")
            yield rgb
            target_index += 1
            if target_index == len(timestamps):
                return
    raise ValueError(f"video ended before all selected frames were decoded: {path}")


def compatible_features():
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"], "fps": 10.0},
        "action": {"dtype": "float32", "shape": (7,), "names": ["actions"], "fps": 10.0},
    }
    for camera in ("image", "image2"):
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": (256, 256, 3),
            "names": ["height", "width", "channel"], "fps": 10.0}
    return features


def validate_export(source, output, selections):
    import av

    if (output.num_episodes != source.num_episodes or output.fps != 10
            or len(output) != sum(map(len, selections))
            or output.meta.info["splits"] != source.meta.info["splits"]
            or not output.meta.tasks.equals(source.meta.tasks)):
        raise ValueError("output episode/task/split metadata mismatch")
    videos = {}
    for episode_index, keep in enumerate(selections):
        old, new = episode_data(source, episode_index), episode_data(output, episode_index)
        for key in ("action", "observation.state", "task_index"):
            if not np.array_equal(new[key], old[key][keep]):
                raise ValueError(f"episode {episode_index}: retained {key} mismatch")
        for camera in output.meta.video_keys:
            path = output.root / output.meta.get_video_file_path(episode_index, camera)
            start = videos.get(path, 0)
            meta = output.meta.episodes[episode_index]
            if (abs(meta[f"videos/{camera}/from_timestamp"] - start / 10) > 1e-3
                    or abs(meta[f"videos/{camera}/to_timestamp"] - (start + len(keep)) / 10) > 1e-3):
                raise ValueError(f"episode {episode_index}: video offsets mismatch")
            videos[path] = start + len(keep)
    for path, expected_count in videos.items():
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if (stream.codec_context.codec.id != av.Codec("av1", "r").id or float(stream.average_rate) != 10
                    or (stream.width, stream.height) != (256, 256)
                    or stream.codec_context.format.name != "yuv420p" or container.streams.audio):
                raise ValueError(f"output video format mismatch: {path}")
            count = 0
            for frame in container.decode(stream):
                if frame.time is None or abs(frame.time - count / 10) > 1e-3:
                    raise ValueError(f"output video timestamps mismatch: {path}")
                count += 1
            if count != expected_count:
                raise ValueError(f"output video frame count mismatch: {path}: {count} != {expected_count}")


def convert(args):
    if version("lerobot") != "0.5.1":
        raise RuntimeError("use lerobot==0.5.1; other writer versions have not been validated")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.io_utils import write_info, write_tasks

    source_root, destination = args.source.resolve(), args.out.resolve()
    pending = destination.with_name(destination.name + ".partial")
    if destination.exists() or pending.exists():
        raise FileExistsError("output or .partial directory exists; choose a new --out")
    if source_root == destination or source_root in destination.parents or destination in source_root.parents:
        raise ValueError("source and output must be separate directories")
    source_info = json.loads((source_root / "meta/info.json").read_text())
    cameras = [args.agent_camera, args.wrist_camera]
    if len(set(cameras)) != 2:
        raise ValueError("agent and wrist cameras must differ")
    expected = DEFAULT_COLUMNS | {"action", "observation.state", *cameras}
    features = source_info["features"]
    if (source_info["codebase_version"] != "v3.0" or source_info["fps"] not in (10, 20)
            or source_info["robot_type"] != "panda" or set(features) != expected
            or features["action"]["shape"] != [7] or features["action"]["dtype"] != "float32"
            or features["observation.state"]["shape"] != [8]
            or features["observation.state"]["dtype"] != "float32"):
        raise ValueError("requires LeRobot v3, panda, 10/20 FPS, float32 state[8]/action[7], two cameras; "
                         "extra features are rejected to avoid silent data loss")
    for camera in cameras:
        if features[camera]["dtype"] != "video" or features[camera]["shape"] != [256, 256, 3]:
            raise ValueError(f"requires 256x256 RGB video: {camera}")

    source = LeRobotDataset("local/source", root=source_root, video_backend="pyav")
    if (source.num_episodes == 0 or len(source.meta.episodes) != source.num_episodes
            or sum(ep["length"] for ep in source.meta.episodes) != len(source)
            or source_info["total_frames"] != len(source)):
        raise ValueError("empty dataset or inconsistent episode/frame totals")
    if not np.array_equal(source.meta.tasks["task_index"].to_numpy(), np.arange(len(source.meta.tasks))):
        raise ValueError("task table must be ordered by contiguous task_index")
    selections = []
    for i in range(source.num_episodes):
        keep = openvla_keep_indices(episode_data(source, i)["action"])
        if len(keep) == 0:
            raise ValueError(f"episode {i}: all actions are no-ops; no output written. "
                             "Resolve this episode explicitly to avoid silently changing tasks/splits")
        selections.append(keep)
    install_safe_image_statistics()
    # The streaming encoder omits image statistics for single-frame episodes.
    streaming = all(len(keep) >= 2 for keep in selections)
    output = LeRobotDataset.create(
        "local/libero_10fps_noops_filtered", root=pending, fps=10, robot_type="panda",
        features=compatible_features(), video_backend="pyav", vcodec="libsvtav1",
        image_writer_threads=2, streaming_encoding=streaming,
        encoder_queue_maxsize=max(map(len, selections)), encoder_threads=2)
    output.meta.tasks = source.meta.tasks.copy()
    write_tasks(output.meta.tasks, pending)
    lineage = []
    try:
        for i, keep in enumerate(selections):
            data = episode_data(source, i)
            meta = source.meta.episodes[i]
            with ExitStack() as stack:
                readers = []
                for camera in cameras:
                    path = source_root / source.meta.get_video_file_path(i, camera)
                    timestamps = data["timestamp"][keep].astype(np.float64) + meta[f"videos/{camera}/from_timestamp"]
                    readers.append(stack.enter_context(closing(selected_video_frames(path, timestamps, source.fps,
                                                                                     features[camera]["shape"]))))
                for step in keep:
                    output.add_frame({
                        "task": source.meta.tasks.iloc[int(data["task_index"][step])].name,
                        "observation.state": data["observation.state"][step], "action": data["action"][step],
                        "observation.images.image": next(readers[0]),
                        "observation.images.image2": next(readers[1])})
                output.save_episode(parallel_encoding=False)
            lineage.append({"episode_index": i, "source_frame_indices": keep.tolist(),
                            "source_length": len(data["action"]), "output_length": len(keep)})
            print(f"Episode {i}: {len(data['action'])} -> {len(keep)} frames", flush=True)
    finally:
        output.finalize()

    # The writer resets splits to train after each episode; episode IDs themselves are preserved.
    output.meta.info["splits"] = source_info["splits"]
    write_info(output.meta.info, pending)
    for name in ("episode_keys.json", "episode_task_map.json", "train_episodes.json", "eval_episodes.json",
                 "validation_episodes.json", "test_episodes.json"):
        if (source_root / name).is_file():
            shutil.copyfile(source_root / name, pending / name)
    if (source_root / "provenance.json").is_file():
        shutil.copyfile(source_root / "provenance.json", pending / "source_provenance.json")
    report = {"source_info_sha256": hashlib.sha256((source_root / "meta/info.json").read_bytes()).hexdigest(),
              "source_fps": source.fps, "output_fps": 10, "time_conversion": "retime_retained_steps",
              "no_op_rule": "openvla-oft", "no_op_source": OPENVLA_SOURCE, "threshold": 1e-4,
              "simulator_replay": False, "image_transform": "identity", "actions_modified": False,
              "versions": {"lerobot": version("lerobot"), "av": version("av"), "numpy": np.__version__},
              "camera_mapping": dict(zip(cameras, output.meta.video_keys)),
              "source_frames": len(source), "output_frames": sum(map(len, selections)),
              "removed_frames": len(source) - sum(map(len, selections)), "episodes": lineage}
    (pending / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    validate_export(source, LeRobotDataset("local/output", root=pending, video_backend="pyav"), selections)
    pending.rename(destination)
    print(f"Saved {len(selections)} episodes, {report['output_frames']} frames; "
          f"removed {report['removed_frames']} no-ops. Output: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="LeRobot v3 root, HDF5 episode, or HDF5 tree")
    parser.add_argument("--out", type=Path, required=True, help="new output directory; source stays intact")
    parser.add_argument("--input-format", choices=["auto", "hdf5", "lerobot"], default="auto")
    parser.add_argument("--relative-actions", action="store_true", required=True,
                        help="confirm raw relative OSC actions and pre-action observations; LeRobot requires -1/+1 gripper")
    parser.add_argument("--agent-camera", default="observation.images.image")
    parser.add_argument("--wrist-camera", default="observation.images.image2")
    parser.add_argument("--pattern", default="preaction.hdf5", help="HDF5 filename pattern, recursively matched")
    parser.add_argument("--actions-key", default="actions")
    parser.add_argument("--state-key", default="observation_state")
    parser.add_argument("--camera-keys", nargs=2, help="HDF5 camera keys; inferred only if exactly two RGB arrays match")
    parser.add_argument("--time-keys", nargs="*", default=[], help="additional HDF5 datasets aligned with actions")
    parser.add_argument("--static-keys", nargs="*", default=[], help="HDF5 datasets that must stay unfiltered")
    parser.add_argument("--fps", type=float, help="source FPS, required only if absent from HDF5 metadata")
    parser.add_argument("--action-offset", type=int, choices=[0], help="confirm pre-action alignment if metadata is absent")
    args = parser.parse_args()
    hdf5 = args.input_format == "hdf5" or (args.input_format == "auto"
                                             and not (args.source / "meta/info.json").is_file())
    (convert_hdf5 if hdf5 else convert)(args)
