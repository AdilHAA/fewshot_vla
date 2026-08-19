"""Shared frozen-encoder helper: trajectory frames -> token latents.

Used by BOTH the offline cache builder and (in P3) the eval live path, so the
exact resize/normalize/dtype pipeline is identical offline and online. The
parent policy's single-frame `_dino_features` is a separate legacy path and is
NOT touched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# How a record's clip is trimmed in time, and how a too-short clip is padded up
# to the encoder's minimum. "all" reproduces every pre-existing cache exactly.
TIME_SELECTS = ("all", "first")
FILLS = ("", "dup", "zero")


def encoder_format(encoder_type: str, vjepa_grid: int = 2, time_select: str = "all",
                   n_frames: int = 0, fill: str = "", dino_grid: int = 0,
                   include_cls: bool = True, n_reg: int = 0) -> str:
    """Cache-header format tag. Describes WHICH tokens a record holds per unit
    (frame for dino, tubelet for vjepa2) and over WHAT temporal scope.

    A 224px frame is cut into 224/14 = 16x16 = 256 patches, so DINOv2 emits
    [CLS, reg x n, patch_0 .. patch_255]. The tag says which of those we keep:

    dino  : "cls"          -> 1 token/frame  (CLS only; the legacy caches)
            "cls+patches"  -> 257            (CLS + ALL 256 patches, nothing pooled)
            "cls+reg{n}"   -> 1+n            (CLS + the n registers of a
                                              dinov2-with-registers checkpoint)
            "cls+grid{s}"  -> 1+s²           (patch map block-averaged to s x s;
                                              s=16 would be a 1x1 block = identity,
                                              which is exactly "patches")
    vjepa2: "tubelet_grid{s}" (s=1 is the legacy full mean-pool).

    With the defaults the tag is byte-for-byte the legacy one, so old caches and
    old configs keep matching. A temporal subselection appends an "@" scope suffix
    ("tubelet_grid1@first1dup" = frame 0 duplicated into one tubelet); that suffix
    is what stops a 1-token cache from being silently accepted by a full-clip
    config. "@" is used because "_" already occurs inside the base tag.
    """
    if encoder_type == "dino":
        if int(dino_grid) < 0 or int(n_reg) < 0:
            raise ValueError(f"dino_grid/n_reg must be >= 0, got {dino_grid}/{n_reg}")
        # `grid16` at 224px pools 1x1 blocks, i.e. does nothing — call that what it
        # is. One canonical name per format, so tags stay comparable as strings.
        patch_tag = ("patches" if int(dino_grid) == DINO_SIDE
                     else f"grid{int(dino_grid)}")
        parts = (["cls"] if include_cls else []) \
            + ([f"reg{int(n_reg)}"] if int(n_reg) else []) \
            + ([patch_tag] if int(dino_grid) else [])
        if not parts:
            raise ValueError("dino format selects no tokens at all")
        base = "+".join(parts)
    elif encoder_type == "vjepa2":
        if int(vjepa_grid) < 1:
            raise ValueError(f"vjepa_grid must be >= 1, got {vjepa_grid}")
        if int(dino_grid) or int(n_reg) or not include_cls:
            raise ValueError("dino_grid/n_reg/include_cls apply to dino only")
        base = f"tubelet_grid{int(vjepa_grid)}"
    else:
        raise ValueError(f"unknown encoder_type {encoder_type!r}")

    if time_select not in TIME_SELECTS:
        raise ValueError(f"unknown time_select {time_select!r}; use one of {TIME_SELECTS}")
    if fill not in FILLS:
        raise ValueError(f"unknown fill {fill!r}; use one of {FILLS}")
    if time_select == "all":
        if int(n_frames) or fill:
            raise ValueError("time_select='all' takes no n_frames/fill")
        return base
    if int(n_frames) < 1:
        raise ValueError("time_select='first' requires n_frames >= 1")
    return f"{base}@first{int(n_frames)}{fill}"


def parse_format(tag: str) -> dict:
    """Inverse of `encoder_format`: tag -> {unit, tokens_per_unit, has_cls, n_reg,
    grid, time_select, n_frames, fill}.

    `tokens_per_unit` is the field the temporal position code needs: tokens are
    unit-major, so a token's unit index is `slot // tokens_per_unit`. Deriving it
    from the tag means every cache ever written is self-describing — no rebuild.
    """
    base, _, scope = tag.partition("@")
    time_select, n_frames, fill = "all", 0, ""
    if scope:
        if not scope.startswith("first"):
            raise ValueError(f"unknown temporal scope {scope!r} in format {tag!r}")
        rest = scope[len("first"):]
        digits = "".join(c for c in rest if c.isdigit())
        if not digits:
            raise ValueError(f"no frame count in temporal scope {scope!r}")
        time_select, n_frames, fill = "first", int(digits), rest[len(digits):]
        if fill not in FILLS:
            raise ValueError(f"unknown fill {fill!r} in format {tag!r}")

    if base.startswith("qwen35vl"):                # Qwen3.5 video-token caches
        # unit = temporal pair (temporal_patch_size=2); at the pinned 224px input the
        # tower emits a 14x14 patch map -> 7x7 merged = 49 tokens per pair.
        out = {"unit": "pair", "has_cls": False, "n_reg": 0, "grid": 7,
               "tokens_per_unit": 49}
        scope = base.split("_", 2)                  # qwen35vl_every{k}
        out.update(time_select="all", n_frames=0, fill="")
        return out
    if base.startswith("tubelet_grid"):
        grid = int(base[len("tubelet_grid"):])
        out = {"unit": "tubelet", "has_cls": False, "n_reg": 0, "grid": grid,
               "tokens_per_unit": grid * grid}
    else:
        has_cls, n_reg, grid = False, 0, 0
        for part in base.split("+"):
            if part == "cls":
                has_cls = True
            elif part.startswith("reg"):
                n_reg = int(part[3:])
            elif part == "patches":                 # all 256, nothing pooled
                grid = DINO_SIDE
            elif part.startswith("grid"):
                grid = int(part[4:])
            else:
                raise ValueError(f"unknown token group {part!r} in format {tag!r}")
        out = {"unit": "frame", "has_cls": has_cls, "n_reg": n_reg, "grid": grid,
               "tokens_per_unit": int(has_cls) + n_reg + grid * grid}
    if out["tokens_per_unit"] < 1:
        raise ValueError(f"format {tag!r} selects no tokens")
    out.update(time_select=time_select, n_frames=n_frames, fill=fill)
    return out


def derive_tokens(toks, src: str, dst: str):
    """(L, D) unit-major tokens in format `src` -> the same clip in format `dst`.

    Lets ONE encoder pass write several caches: `cls` is literally token 0 of every
    unit of `cls+patches` / `cls+reg4`, so the cheap cache costs no GPU at all. The
    decode pass is what dominates a build, and this pays it once instead of twice.

    CLS and register slices are BITWISE exact; a grid reduction is exact in real
    arithmetic (a mean of equal-size disjoint means is the mean) and therefore exact
    to fp16 storage. Widening is impossible and raises.
    """
    a, b = parse_format(src), parse_format(dst)
    if a["unit"] != b["unit"] or (a["time_select"], a["n_frames"], a["fill"]) != \
            (b["time_select"], b["n_frames"], b["fill"]):
        raise ValueError(f"cannot derive {dst!r} from {src!r}: different unit or scope")
    if (b["has_cls"] and not a["has_cls"]) or b["n_reg"] > a["n_reg"]:
        raise ValueError(f"cannot derive {dst!r} from {src!r}: missing cls/registers")
    if b["grid"] and (not a["grid"] or a["grid"] % b["grid"]):
        raise ValueError(f"cannot derive {dst!r} from {src!r}: grid {a['grid']} -> {b['grid']}")

    t = torch.as_tensor(toks)
    n_units, d = t.shape[0] // a["tokens_per_unit"], t.shape[-1]
    u = t.reshape(n_units, a["tokens_per_unit"], d)
    parts = []
    if b["has_cls"]:
        parts.append(u[:, :1])
    if b["n_reg"]:
        parts.append(u[:, int(a["has_cls"]):int(a["has_cls"]) + b["n_reg"]])
    if b["grid"]:
        patches = u[:, int(a["has_cls"]) + a["n_reg"]:]
        parts.append(patches if b["grid"] == a["grid"]
                     else grid_pool(patches.float(), b["grid"]).to(t.dtype))
    out = torch.cat(parts, dim=1).reshape(n_units * b["tokens_per_unit"], d)
    return out.numpy() if not torch.is_tensor(toks) else out


def grid_pool(tok: torch.Tensor, grid: int) -> torch.Tensor:
    """(X, side*side, D) spatial map -> (X, grid*grid, D) by BLOCK averaging.

    Block (not strided) pooling is what makes the format hierarchy derivable:
    pooling a grid=4 map down to 2×2 equals pooling the original 16×16 map to 2×2,
    because a mean of equal-size disjoint means is the mean. Row-major order."""
    x, n_sp, d = tok.shape
    side = int(n_sp ** 0.5)
    if side * side != n_sp:
        raise ValueError(f"spatial map {n_sp} is not square")
    if grid < 1 or side % grid:
        raise ValueError(f"spatial map {side}×{side} not divisible into {grid}×{grid}")
    t = tok.reshape(x, grid, side // grid, grid, side // grid, d)
    return t.mean(dim=(2, 4)).reshape(x, grid * grid, d)


def imagenet_buffers(device, dtype):
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


@torch.no_grad()
def dino_encode(model, mean, std, frames: torch.Tensor, grid: int = 0,
                include_cls: bool = True, n_reg: int = 0) -> torch.Tensor:
    """frames: (B,T,C,H,W) in [0,1] -> (B, T*tokens_per_frame, D) fp16.

    With the defaults this is the legacy path: one CLS per frame, shape (B,T,D).
    `n_reg` takes the register tokens of a dinov2-with-registers checkpoint and
    `grid` block-pools the patch map to grid×grid tokens.

    Token layout of Dinov2WithRegisters is [CLS, reg×N, patches] (registers are
    concatenated AFTER the position embeddings are added, so they carry none), so
    the patch slice MUST start at 1+n_reg — a naive [:, 1:] would feed 4 register
    tokens into the spatial reshape and silently corrupt the grid."""
    b, t, c, h, w = frames.shape
    x = frames.reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - mean) / std
    model_dtype = next(model.parameters()).dtype
    out = model(pixel_values=x.to(model_dtype))
    hs = out.last_hidden_state                            # (B*T, 1+n_reg+n_patch, D)
    if grid == 0 and include_cls and n_reg == 0:          # legacy fast path
        return hs[:, 0].reshape(b, t, -1).to(torch.float16)
    parts = []
    if include_cls:
        parts.append(hs[:, :1])
    if n_reg:
        parts.append(hs[:, 1:1 + n_reg])
    if grid:
        parts.append(grid_pool(hs[:, 1 + n_reg:], grid))
    if not parts:
        raise ValueError("dino_encode selects no tokens (grid=0, include_cls=False, n_reg=0)")
    tok = torch.cat(parts, dim=1)                         # (B*T, tokens_per_frame, D)
    return tok.reshape(b, t * tok.shape[1], -1).to(torch.float16)


DINO_INPUT = 224
DINO_PATCH = 14
DINO_SIDE = DINO_INPUT // DINO_PATCH   # 16x16 = 256 patches per frame

VJEPA2_SIZE = 256
VJEPA2_PATCH = 16
VJEPA2_TUBELET = 2
# Frames a record physically needs: dino is per-frame, V-JEPA2's patch embed is a
# Conv3d of depth `tubelet_size`.
MIN_FRAMES = {"dino": 1, "vjepa2": VJEPA2_TUBELET}


def check_vjepa2_geometry(model) -> None:
    """Fail fast if the checkpoint's patch geometry differs from what our fixed
    `VJEPA2_SIZE` resize assumes.

    V-JEPA2 stores no position embeddings: it derives the 3D-RoPE grid from
    `config.crop_size // config.patch_size` and takes a token's temporal id as
    `index // grid_size**2`. Feeding a 256²-resized clip to a checkpoint whose
    config says 224² therefore assigns SEVERAL DISTINCT temporal positions inside
    one tubelet — silently, with no shape error. Test stubs carry no config and
    are skipped.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return
    # Sentinel None, not the expected value: a config that simply lacks these
    # fields must fail loudly rather than be waved through as "matching".
    got = tuple(getattr(cfg, k, None)
                for k in ("crop_size", "patch_size", "tubelet_size"))
    if None in got:
        raise ValueError(
            f"vjepa2 config {type(cfg).__name__} lacks crop_size/patch_size/"
            f"tubelet_size (got {got}); cannot verify the 3D-RoPE grid")
    got = tuple(int(v) for v in got)
    want = (VJEPA2_SIZE, VJEPA2_PATCH, VJEPA2_TUBELET)
    if got != want:
        raise ValueError(
            f"vjepa2 checkpoint geometry (crop_size, patch_size, tubelet_size)={got} "
            f"!= expected {want}; the fixed {VJEPA2_SIZE}px resize would corrupt the "
            "3D-RoPE temporal ids")


@torch.no_grad()
def vjepa2_encode(model, mean, std, frames: torch.Tensor, grid: int = 2) -> torch.Tensor:
    """frames: (B,T,C,H,W), T even and >= 2 -> (B, (T//2)·grid², D) fp16.

    V-JEPA2 has no CLS; each tubelet's 16×16 spatial map is average-pooled into a
    grid×grid block grid (grid=1 reproduces the legacy full mean-pool). Tokens are
    ordered tubelet-major, row-major within the grid."""
    b, t, c, h, w = frames.shape
    if grid < 1:
        raise ValueError(f"vjepa2 grid must be >= 1, got {grid}")
    # Explicit raise, not `assert`: under `python -O` an assert vanishes, and then
    # T=1 dies with ZeroDivisionError below while odd T silently DROPS the last
    # frame (the Conv3d depth stride yields floor(T/2) tubelets, no error).
    if t < VJEPA2_TUBELET or t % VJEPA2_TUBELET:
        raise ValueError(
            f"vjepa2 needs an even frame count >= {VJEPA2_TUBELET} "
            f"(tubelet_size={VJEPA2_TUBELET}), got T={t}")
    check_vjepa2_geometry(model)
    x = frames.reshape(b * t, c, h, w)
    x = F.interpolate(x, size=(VJEPA2_SIZE, VJEPA2_SIZE), mode="bilinear", align_corners=False)
    x = ((x - mean) / std).reshape(b, t, c, VJEPA2_SIZE, VJEPA2_SIZE)
    model_dtype = next(model.parameters()).dtype
    tok = model.get_vision_features(x.to(model_dtype))    # (B, (T//2)*n_sp, D)
    n_tub = t // VJEPA2_TUBELET
    n_sp, d = tok.shape[1] // n_tub, tok.shape[-1]
    tok = grid_pool(tok.reshape(b * n_tub, n_sp, d), grid)
    return tok.reshape(b, n_tub * grid * grid, -1).to(torch.float16)


def expand_clip(frames: torch.Tensor, min_frames: int, fill: str = "dup") -> torch.Tensor:
    """frames: (T,C,H,W) -> (T',C,H,W) with T' >= min_frames.

    `dup` repeats the last frame — for a single frame this yields [f0, f0], whose
    tubelet is exactly the 2D convolution (W0+W1)·f0, i.e. motion is removed by
    construction while the frame content is kept whole. `zero` pads with black
    frames instead and exists only as a measured-worse control: a black second
    frame moves the tubelet further than an unrelated image does.
    """
    # Validated before the early return, so a typo cannot pass silently on the one
    # path where no padding happens (dino, min_frames=1).
    if fill not in ("dup", "zero"):
        raise ValueError(f"unknown fill {fill!r}; use 'dup' or 'zero'")
    t = int(frames.shape[0])
    if t < 1:
        raise ValueError(f"expand_clip needs at least one frame, got T={t}")
    if t >= min_frames:
        return frames
    pad = min_frames - t
    if fill == "dup":
        extra = frames[-1:].expand(pad, *frames.shape[1:])
    else:                                                 # device= matters: the
        extra = torch.zeros(pad, *frames.shape[1:],       # frames may be on cuda
                            dtype=frames.dtype, device=frames.device)
    return torch.cat([frames, extra], dim=0)


_DEFAULT_MODEL_ID = {"dino": "facebook/dinov2-base", "vjepa2": "facebook/vjepa2-vitl-fpc64-256"}


def default_model_id(encoder_type: str) -> str:
    """HF id used when none is given — recorded in the cache header, because
    `encoder_id` ("dino") does NOT identify the weights: dinov2-base and
    dinov2-with-registers-base otherwise produce byte-identical headers."""
    if encoder_type not in _DEFAULT_MODEL_ID:
        raise ValueError(f"unknown encoder_type {encoder_type!r}; use 'dino' or 'vjepa2'")
    return _DEFAULT_MODEL_ID[encoder_type]


def build_traj_encoder(encoder_type: str, model_id: str | None = None,
                       device="cpu", dtype=torch.float32, vjepa_grid: int = 2,
                       dino_grid: int = 0, include_cls: bool = True, n_reg: int = 0,
                       model_dtype=None):
    """Load a frozen clip encoder and return (model, encode_fn). encode_fn maps
    frames (B,T,C,H,W) in [0,1] -> (B, N, D) fp16. Shared by the offline builder
    and the live eval path so preprocessing is identical.

    `model_dtype=None` picks fp16 on CUDA and fp32 on CPU. This used to be fp32
    everywhere — the `dtype` argument only ever reached the ImageNet buffers — which
    cost a measured 2.76x on the encode pass for no benefit: the tokens are cast to
    fp16 on the way out regardless."""
    from transformers import AutoModel

    if encoder_type not in _DEFAULT_MODEL_ID:             # guard also when model_id
        raise ValueError(f"unknown encoder_type {encoder_type!r}; "  # is given
                         "use 'dino' or 'vjepa2'")
    if model_dtype is None:
        model_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(model_id or default_model_id(encoder_type),
                                      dtype=model_dtype)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    mean, std = imagenet_buffers(device, dtype)
    if encoder_type == "dino":
        return model, (lambda frames: dino_encode(model, mean, std, frames.to(device),
                                                  grid=dino_grid, include_cls=include_cls,
                                                  n_reg=n_reg))
    check_vjepa2_geometry(model)
    return model, (lambda frames: vjepa2_encode(model, mean, std, frames.to(device),
                                                grid=vjepa_grid))
