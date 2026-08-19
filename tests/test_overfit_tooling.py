"""Tests for the direction-6 tooling: pick_task (task->episodes) and the loss-slope
gate. Headless; the TB part writes real event files via the tensorboard protobuf
writer. Run: python3 tests/test_overfit_tooling.py
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.loss_slope import analyse, slope
from scripts.pick_task import task_table
from src.traj_data.cache_io import CacheHeader, write_cache


def _toy_cache(d):
    """10 episodes over 4 tasks; record length encodes the episode index."""
    recs, seqs = [], []
    for ep in range(10):
        n = 100 + ep * 10
        seqs.append(np.zeros((n, 4), np.float16))
        recs.append({"episode": ep, "variant": 0, "task_index": ep // 3})
    # one augmented variant must NOT pollute the pool
    seqs.append(np.zeros((50, 4), np.float16))
    recs.append({"episode": 0, "variant": 2, "task_index": 0})
    hdr = CacheHeader("dino", "cls", 4, "orig", len(recs), tokens_per_unit=1)
    write_cache(d, seqs, recs, hdr, {0: "open the box", 1: "close the box"})
    return hdr


def test_task_table_originals_only_and_frames():
    with tempfile.TemporaryDirectory() as d:
        _toy_cache(d)
        import json
        meta = json.load(open(os.path.join(d, "index.json")))
        t = task_table(meta["records"], meta["header"])
        assert sorted(t) == [0, 1, 2, 3]
        assert t[0]["episodes"] == [0, 1, 2]          # variant-2 row excluded
        # 'cls' keeps 1 token per frame -> frames == record lengths
        assert t[0]["frames"] == 100 + 110 + 120
        assert t[3]["episodes"] == [9]


def test_task_table_tubelet_frames():
    """tubelet_grid2: 4 tokens cover 2 frames -> per_frame = 4/2 = 2."""
    with tempfile.TemporaryDirectory() as d:
        seqs = [np.zeros((40, 8), np.float16)]
        recs = [{"episode": 0, "variant": 0, "task_index": 5}]
        write_cache(d, seqs, recs,
                    CacheHeader("vjepa2", "tubelet_grid2", 8, "orig", 1,
                                tokens_per_unit=4), {})
        import json
        meta = json.load(open(os.path.join(d, "index.json")))
        t = task_table(meta["records"], meta["header"])
        assert t[5]["frames"] == 20.0                  # 40 tokens / (4/2)


def test_slope_and_plateau_verdict():
    assert slope([0, 1, 2, 3], [3, 2, 1, 0], 0, 10)[0] == -1.0
    m, n = slope([0, 1], [0, 0], 5, 10)
    assert m is None and n == 0                        # empty window

    import time

    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "tensorboard"))
        w = EventFileWriter(os.path.join(d, "tensorboard"))

        def scalar(tag, val, step):
            e = Event(wall_time=time.time(), step=step)
            v = e.summary.value.add(); v.tag = tag; v.simple_value = val
            w.add_event(e)

        for step in range(0, 100_000, 500):
            v = max(0.20, 1.0 - 0.8 * (step / 30_000)) if step < 30_000 else 0.2005
            scalar("train/loss", v, step)
            scalar("train/lr", 1e-4 if step < 30_000 else 2.5e-6, step)
        w.close()

        text, ok = analyse(d, decay=30_000)
        assert ok
        assert "plateau" in text and "2.50e-06" in text

        # and the improving case: loss keeps falling through both windows
        with tempfile.TemporaryDirectory() as d2:
            os.makedirs(os.path.join(d2, "tensorboard"))
            w = EventFileWriter(os.path.join(d2, "tensorboard"))
            for step in range(0, 100_000, 500):
                scalar_tag = 1.0 - 0.6 * step / 100_000
                e = Event(wall_time=time.time(), step=step)
                v = e.summary.value.add(); v.tag = "train/loss"; v.simple_value = scalar_tag
                w.add_event(e)
            w.close()
            text2, _ = analyse(d2, decay=30_000)
            assert "still improving" in text2 or "plateau" not in text2


RUN = [test_task_table_originals_only_and_frames, test_task_table_tubelet_frames,
       test_slope_and_plateau_verdict]
if __name__ == "__main__":
    for fn in RUN:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL OVERFIT-TOOLING TESTS PASS")
