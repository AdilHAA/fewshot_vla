"""The prompt layout Qwen3.5 sees for a video, rebuilt from cached tokens.

The trunk arm feeds PRE-ENCODED video tokens straight into the frozen text stack, so
nothing in transformers builds the prompt for us. Feeding a bare token soup discards
two things the checkpoint was trained with: the `<t seconds>` / `<|vision_start|>`
scaffolding around every temporal patch, and the 3-D M-RoPE positions that make the
7x7 grid of a patch share one temporal index instead of marching 49 steps forward.
These helpers reproduce both EXACTLY as Qwen3_5Model does, and are pure
torch/python (no transformers import, no weights) so the replication is testable
against the reference implementation without a GPU.

Reference: transformers 5.7.0, models/qwen3_5/modeling_qwen3_5.py and
models/qwen3_vl/processing_qwen3_vl.py; line numbers are cited per function.

Grid units: `video_grids` are PATCH counts (what the processor reports as
`video_grid_thw`), NOT token counts — a 224x224 frame pair is (1, 14, 14) patches
and becomes 7x7 = 49 tokens after the merger's spatial_merge_size=2.
"""
from __future__ import annotations

import itertools

import torch
from torch import Tensor

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
VIDEO_PAD = "<|video_pad|>"

PAD_MULTIPLE = 256


def build_prompt_text(demo_timestamps: list[list[float]], instruction: str | None,
                      tokens_per_unit: int = 49) -> str:
    """The chat string a Qwen3.5 video turn would be: one `<t seconds>`-labelled
    vision block per temporal patch, then the instruction, then the assistant header
    with the empty think block the checkpoint's chat template always emits under
    add_generation_prompt (enable_thinking=False) — the layer queries then sit where
    the answer would start, the one position the model has seen after such a prompt.

    Qwen3VLProcessor expands a single `<|vision_start|><|video_pad|><|vision_end|>`
    into exactly this per-frame form (processing_qwen3_vl.py:158-166), so writing it
    out directly is the same string the processor would have produced — we only skip
    the pixel path, because the tokens are already cached.

    Several demos are concatenated inside ONE user turn: they are alternatives of the
    same task, and the trunk is meant to attend across them.
    """
    parts = [f"{IM_START}user\n"]
    for stamps in demo_timestamps:
        for t in stamps:
            parts.append(f"<{t:.1f} seconds>{VISION_START}"
                         f"{VIDEO_PAD * tokens_per_unit}{VISION_END}")
    if instruction:
        parts.append(instruction)
    parts.append(f"{IM_END}\n{IM_START}assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


def _vision_position_ids(start_position: int, grid_thw: tuple[int, int, int],
                         spatial_merge_size: int) -> Tensor:
    """Port of Qwen3_5Model.get_vision_position_ids (modeling_qwen3_5.py:1341-1397)
    with temp_merge_size=1 and time_interval=1, the values get_rope_index passes."""
    t, h, w = (int(grid_thw[0]), int(grid_thw[1] // spatial_merge_size),
               int(grid_thw[2] // spatial_merge_size))
    pos_t = torch.arange(t)
    pos_w = (torch.arange(w) + start_position).repeat(h * t)
    pos_h = (torch.arange(h) + start_position).repeat_interleave(w).repeat(t)
    pos_t = pos_t.repeat_interleave(h * w) + start_position
    return torch.stack([pos_t, pos_h, pos_w], dim=0)


def mrope_positions(token_types: Tensor, video_grids: list[tuple[int, int, int]],
                    spatial_merge_size: int = 2) -> Tensor:
    """(3, L) M-RoPE positions for ONE unpadded row — port of
    Qwen3_5Model.get_rope_index (modeling_qwen3_5.py:1399-1490), batch size 1.

    token_types: 0 = text, 2 = video (`mm_token_type_ids`; only the `<|video_pad|>`
    slots are 2 — the timestamp text and the vision_start/end markers are text, see
    ProcessorMixin.create_mm_token_type_ids, processing_utils.py:1657-1669).
    video_grids: one (T, H, W) in patch units per video block, in order.

    The two rules worth naming, because they are what a naive `arange` gets wrong:
    a video whose grid says T frames is FIRST split into T grids of T=1 (line 1447 —
    Qwen3.5 separates frames with timestamps, so each frame is its own block), and a
    block advances the running position by only max(H, W) // merge (line 1483), not
    by its 49 tokens.
    """
    types = [int(v) for v in token_types.tolist()]
    grids: list[tuple[int, int, int]] = []
    for t, h, w in video_grids:                    # get_rope_index:1446-1448
        grids.extend([(1, int(h), int(w))] * int(t))
    grid_iter = iter(grids)

    pos = torch.zeros(3, len(types), dtype=torch.long)
    cur = 0
    for key, group in itertools.groupby(enumerate(types), lambda x: x[1]):
        group = list(group)
        start, end = group[0][0], group[-1][0] + 1
        if key == 0:
            n = end - start
            pos[:, start:end] = torch.arange(n).view(1, -1).expand(3, -1) + cur
            cur += n
        elif key == 2:
            grid = next(grid_iter)
            block = _vision_position_ids(cur, grid, spatial_merge_size)
            if block.shape[1] != end - start:
                raise ValueError(
                    f"video block of {end - start} tokens does not match grid {grid} "
                    f"at merge {spatial_merge_size} ({block.shape[1]} tokens)")
            pos[:, start:end] = block
            cur += max(grid[1], grid[2]) // spatial_merge_size
        else:
            raise ValueError(f"unsupported token type {key} (expected 0 text / 2 video)")
    if next(grid_iter, None) is not None:
        raise ValueError("more video grids than video blocks in token_types")
    return pos


def four_row_positions(pos3: Tensor) -> Tensor:
    """(3, L) M-RoPE -> the (4, L) tensor Qwen3_5TextModel.forward expects.

    FINDING (modeling_qwen3_5.py:1257-1269): "the hard coded `4` is for text,
    temporal, height and width" — a 3-D position_ids is split as
    `text_position_ids = position_ids[0]` / mrope = `position_ids[1:]` ONLY when it
    has 4 rows; anything else (None, 2-D, or the 3-row tensor Qwen3_5Model.forward
    itself hands down at line 1685 from compute_3d_position_ids) leaves
    text_position_ids = None. So the reference video path never fills row 0.

    Row 0 is inert for us: it reaches only create_causal_mask (line 1276), which
    consults position_ids purely to detect packed sequences and only when
    attention_mask is None (masking_utils.py:871), and the attention's unused
    **kwargs (line 1290). We always pass an attention_mask, so 4 rows are
    bit-equivalent to the reference's 3 — row 0 gets the plain 1-D positions the
    model would build for itself (line 1259), which keeps the tensor self-describing
    if a future layer starts reading it.
    """
    if pos3.shape[0] != 3:
        raise ValueError(f"expected (3, ...) mrope positions, got {tuple(pos3.shape)}")
    text = torch.arange(pos3.shape[-1], dtype=pos3.dtype, device=pos3.device)
    text = text.expand(pos3.shape[1:])
    return torch.cat([text.unsqueeze(0), pos3], dim=0)


def left_pad_rows(rows: list[tuple[Tensor, Tensor | None]]):
    """[(embeds (L,d), pos4 (4,L))] -> (seq (B,Lmax,d), attn (B,Lmax) bool, pos (4,B,Lmax)).

    LEFT padding keeps the layer queries at a batch-wide tail offset, and Lmax is
    rounded up to a multiple of 256 so the trunk's Triton kernels stop recompiling on
    every new sequence length. Pad slots are zero in both the embeddings and the
    positions and masked out of attention. A single row is not padded at all:
    Qwen3.5's apply_mask_to_padding_states zeroes pad hidden states only when the
    batch has more than one row, so a padded lone row would leak them into the
    linear-attention state.

    pos4 may be None on every row (the raw layout has no explicit positions and lets
    the trunk build its own 1-D arange); the returned positions are then None too.
    """
    if not rows:
        raise ValueError("no rows to pad")
    lengths = [e.shape[0] for e, _ in rows]
    no_pos = all(p is None for _, p in rows)
    for (e, p), n in zip(rows, lengths):
        if not no_pos and (p is None or p.shape != (4, n)):
            raise ValueError(f"positions {None if p is None else tuple(p.shape)} do "
                             f"not match {n} embeddings")
    lmax = max(lengths)
    if len(rows) > 1:
        lmax = ((lmax + PAD_MULTIPLE - 1) // PAD_MULTIPLE) * PAD_MULTIPLE
    e0 = rows[0][0]
    seq = torch.zeros(len(rows), lmax, e0.shape[1], dtype=e0.dtype, device=e0.device)
    attn = torch.zeros(len(rows), lmax, dtype=torch.bool, device=e0.device)
    pos = None if no_pos else torch.zeros(4, len(rows), lmax, dtype=torch.long,
                                          device=e0.device)
    for b, ((emb, p4), n) in enumerate(zip(rows, lengths)):
        seq[b, lmax - n:] = emb
        attn[b, lmax - n:] = True
        if pos is not None:
            pos[:, b, lmax - n:] = p4.to(pos.device)
    return seq, attn, pos
