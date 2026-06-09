"""Synthesize LIBERO-Pro object-axis variants from the stock LIBERO objects.

The ``_object`` perturbation renames the movable objects to visually-changed
variants — ``bigger_*`` (scaled up) and ``<color>_*`` (recolored, e.g.
``red_basket``, ``green_butter``). LIBERO-Pro ships only bddl/init files (no
assets), so those category names are absent from LIBERO's ``OBJECTS_DICT`` and
the env build raises ``KeyError``.

We rebuild each variant from its base object by mutating the already-parsed
MuJoCo XML in memory (robosuite resolves asset paths to absolute on load, so no
files are copied): recolor overrides the textured material's ``rgba``; scale
multiplies the mesh ``scale`` plus every collision geom ``pos``/``size`` and
site ``pos`` so collisions and placement stay consistent.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from libero.libero.envs.base_object import OBJECTS_DICT

logger = logging.getLogger(__name__)

# modifier prefix -> solid rgba that replaces the textured material
_COLORS: dict[str, str] = {
    "red": "0.70 0.10 0.10 1",
    "green": "0.10 0.55 0.15 1",
    "blue": "0.12 0.20 0.80 1",
    "yellow": "0.88 0.80 0.10 1",
    "white": "0.92 0.92 0.92 1",
    "black": "0.08 0.08 0.08 1",
    "brown": "0.40 0.26 0.12 1",
}
# modifier prefix -> uniform scale factor
_SCALE: dict[str, float] = {"bigger": 1.35, "smaller": 0.7}
_MODS: tuple[str, ...] = tuple([*_COLORS, *_SCALE])

# Irregular variants whose base/modifier the prefix rule can't recover
# (the base object is named differently from the variant stem).
_ALIASES: dict[str, tuple[str, str | None]] = {
    "white_bottle": ("wine_bottle", "white"),
    "yellow_cabinet": ("wooden_cabinet", "yellow"),
    "yellow_stove": ("flat_stove", "yellow"),
    # book→bowl substitution in libero_10_object: best-effort, no visual change
    "black_bowl": ("akita_black_bowl", None),
}

_OBJ_BLOCK = re.compile(r"\(:(?:objects|fixtures)(.*?)\)", re.DOTALL)
_OBJ_LINE = re.compile(r"^\s*\S+\s+-\s+(\S+)\s*$")

# Arena surfaces are resolved by the scene/arena, not via OBJECTS_DICT.
_ARENA_SURFACES = {"floor", "table", "kitchen_table", "living_room_table", "study_table"}


def _mul_vec(s: str, f: float) -> str:
    return " ".join(f"{float(x) * f:.6g}" for x in s.split())


def _apply_color(obj, rgba: str) -> None:
    for mat in obj.asset.iter("material"):
        mat.attrib.pop("texture", None)  # solid color, drop the texture map
        mat.set("rgba", rgba)
    for g in obj.worldbody.iter("geom"):
        if g.get("group") == "1" and not g.get("material") and g.get("rgba"):
            g.set("rgba", rgba)  # textureless visual geom: recolor directly


def _apply_scale(obj, f: float) -> None:
    for m in obj.asset.iter("mesh"):
        if m.get("scale"):
            m.set("scale", _mul_vec(m.get("scale"), f))
    for g in obj.worldbody.iter("geom"):
        for attr in ("pos", "size"):
            if g.get(attr):
                g.set(attr, _mul_vec(g.get(attr), f))
    for s in obj.worldbody.iter("site"):
        if s.get("pos"):
            s.set("pos", _mul_vec(s.get("pos"), f))


def _resolve(key: str) -> tuple[str | None, str | None]:
    """(base_key, modifier) for a variant key, or (None, None) if unresolvable.

    modifier is None for a pure alias (register base under a new name, no change).
    """
    if key in _ALIASES:
        base, mod = _ALIASES[key]
        return (base, mod) if base in OBJECTS_DICT else (None, None)
    for mod in _MODS:
        stem = key[len(mod) + 1 :]
        if key.startswith(mod + "_") and stem in OBJECTS_DICT:
            return stem, mod
    return None, None


def _make_variant(base_key: str, mod: str | None):
    base_cls = OBJECTS_DICT[base_key]

    if mod in _SCALE:
        factor = _SCALE[mod]

        def __init__(self, name, **kw):  # noqa: N807
            base_cls.__init__(self, name=name, **kw)
            _apply_scale(self, factor)
    elif mod in _COLORS:
        rgba = _COLORS[mod]

        def __init__(self, name, **kw):  # noqa: N807
            base_cls.__init__(self, name=name, **kw)
            _apply_color(self, rgba)
    else:  # pure alias — no visual change

        def __init__(self, name, **kw):  # noqa: N807
            base_cls.__init__(self, name=name, **kw)

    return type(base_cls)("LiberoProVariant", (base_cls,), {"__init__": __init__})


def register_object_keys(keys) -> list[str]:
    """Register any not-yet-known object keys as synthesized variants. Idempotent."""
    added: list[str] = []
    for key in keys:
        if key in OBJECTS_DICT:
            continue
        base, mod = _resolve(key)
        if base is None:
            logger.warning(
                "LIBERO-Pro: cannot synthesize object %r (unknown base); any "
                "task using it will still KeyError.",
                key,
            )
            continue
        OBJECTS_DICT[key] = _make_variant(base, mod)
        added.append(key)
    if added:
        logger.info("LIBERO-Pro: synthesized %d object variant(s): %s",
                    len(added), ", ".join(sorted(added)))
    return added


def object_categories_in_bddls(bddl_dir: Path) -> set[str]:
    """Movable-object category names declared across a suite's BDDL files."""
    cats: set[str] = set()
    for bddl in bddl_dir.glob("*.bddl"):
        text = bddl.read_text()
        for block in _OBJ_BLOCK.finditer(text):
            for line in block.group(1).splitlines():
                lm = _OBJ_LINE.match(line)
                if lm:
                    cats.add(lm.group(1))
    return cats - _ARENA_SURFACES  # surfaces are arena fixtures, not OBJECTS_DICT
