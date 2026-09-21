"""Single-threaded video decoding for per-sample frame reads.

lerobot reads one frame per training sample through torchvision's pyav
`VideoReader`, i.e. a fresh `av.open()` per sample with default thread settings.
libdav1d (the AV1 decoder) then builds a thread pool sized by the machine's CPU
count on every open — 224 threads on the SR006 node inside a 48-core cgroup quota —
and the first frame after a seek costs ~0.1 s instead of ~0.02 s (measured
2026-09-21 on libero_all: 0.111 -> 0.020 s per frame with thread_count=1; the
DataLoader workers were the training bottleneck, 1.5 step/s vs a ~3.3 step/s GPU
ceiling). Decoded frames are byte-identical: threading changes scheduling, not
output. Applied in train_hyper_lora.py before the DataLoader workers fork.
"""

from __future__ import annotations

import av

_ORIG_ATTR = "_hn_orig_open"


def limit_decoder_threads(threads: int = 1):
    """Patch `av.open` so every video stream of an opened container decodes with
    `threads` threads (0 = leave the stock behaviour, no patch). Idempotent.
    Returns a callable that restores the stock `av.open`."""
    if threads <= 0 or getattr(av.open, _ORIG_ATTR, None) is not None:
        return lambda: None
    orig = av.open

    def patched(*args, **kwargs):
        container = orig(*args, **kwargs)
        for stream in getattr(container.streams, "video", ()):
            stream.thread_type = "NONE"
            stream.thread_count = threads
        return container

    setattr(patched, _ORIG_ATTR, orig)
    av.open = patched

    def restore():
        if av.open is patched:
            av.open = orig

    return restore
