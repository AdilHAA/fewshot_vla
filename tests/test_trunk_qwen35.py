"""Headless tests for the Qwen3.5 trunk (direction 7). The real trunk is replaced by
a tiny causal transformer stub — the assembly, padding, readout and gradient paths
are what's under test; the real-checkpoint behaviour (load, grad-through-frozen,
checkpointing) was verified separately on the actual weights.
Run: python3 tests/test_trunk_qwen35.py
"""
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

for pkgname, rel in [("src", "src"), ("src.hyper_lora", "src/hyper_lora"),
                     ("src.hyper_lora_traj", "src/hyper_lora_traj")]:
    if pkgname not in sys.modules:
        pkg = types.ModuleType(pkgname)
        pkg.__path__ = [os.path.join(ROOT, rel)]
        sys.modules[pkgname] = pkg

from src.hyper_lora_traj.trunk_qwen35 import TrunkHyperNetwork
from src.traj_data.stride import stride_indices, stride_slice

D, HID, NL = 32, 16, 4          # trunk dim / head dim / num layer tokens


class _StubTrunk(torch.nn.Module):
    """Causal transformer standing in for Qwen3_5TextModel: exercises the
    attention-mask polarity for real (True=keep in our convention)."""
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(64, D)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=D, nhead=2, dim_feedforward=2 * D, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.config = types.SimpleNamespace(num_hidden_layers=1, hidden_size=D)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=False):
        # torch: True==IGNORE for key_padding_mask; ours: 1=keep
        out = self.enc(inputs_embeds, src_key_padding_mask=~attention_mask.bool())
        return types.SimpleNamespace(last_hidden_state=out)


class _FakeTok:
    def __call__(self, texts, **kw):
        ids = torch.tensor([[1, 2, 3] for _ in texts])
        return types.SimpleNamespace(input_ids=ids)


def _hn(**kw):
    torch.manual_seed(0)
    hn = TrunkHyperNetwork(
        text_embed_dim=8, hidden_size=HID, num_layers=NL, lora_rank=2, lora_alpha=4,
        target_modules={"m": (8, 8)}, encoder_type="tf", tf_num_blocks=1,
        tf_num_heads=2, zero_init_up=False,
        trunk_model="stub", trunk_dim=D, trunk_q_std=0.5, **kw).eval()
    hn._trunk = _StubTrunk().eval()
    hn._trunk_tokenizer = _FakeTok()
    return hn


def _flat(hn, toks, pad, texts):
    w = hn.forward_trunk(toks, pad, texts)
    return torch.cat([t.flatten() for pl in w.values() for pair in pl.values() for t in pair])


def test_stride_covers_whole_clip_and_is_even():
    idx = stride_indices(140, 32)
    assert len(idx) <= 32 and idx[0] == 0 and idx[-1] == 139   # start AND end kept
    assert len(idx) % 2 == 0                                    # temporal pairs
    assert list(idx) == sorted(set(idx.tolist()))
    assert list(stride_indices(10, 32)) == list(range(10))      # short clip: all
    assert stride_indices(7, 3).shape[0] == 4                   # odd budget rounds up
    # the scratch control slices the SAME frames from the dino cache
    import numpy as np
    cls = np.arange(140)
    assert list(stride_slice(cls, 32)) == list(cls[stride_indices(140, 32)])


def test_output_contract_matches_every_hypernetwork():
    hn = _hn()
    toks = torch.randn(2, 50, D)
    pad = torch.zeros(2, 50, dtype=torch.bool)
    w = hn.forward_trunk(toks, pad, ["open the box", "close the box"])
    assert set(w) == {"m"} and set(w["m"]) == set(range(NL))
    wd, wu = w["m"][0]
    assert wd.shape == (2, 2, 8) and wu.shape == (2, 8, 2)


def test_left_pad_slots_cannot_reach_the_readout():
    """Padded demo slots are dropped at assembly; garbage there must be invisible."""
    hn = _hn()
    toks = torch.randn(2, 50, D)
    pad = torch.zeros(2, 50, dtype=torch.bool)
    pad[0, 30:] = True                        # demo 0 is shorter
    base = _flat(hn, toks, pad, ["a", "b"])
    toks2 = toks.clone()
    toks2[0, 30:] = 1e3                       # poison the pad slots
    assert torch.equal(_flat(hn, toks2, pad, ["a", "b"]), base)


def test_gradients_reach_queries_and_proj_but_not_trunk():
    hn = _hn(trunk_text=False).train()
    hn._trunk.requires_grad_(False)
    toks = torch.randn(2, 40, D)                    # cached tokens are constants
    pad = torch.zeros(2, 40, dtype=torch.bool)
    loss = _flat(hn, toks, pad, ["", ""]).pow(2).sum()
    loss.backward()
    assert hn.layer_queries.grad is not None and hn.layer_queries.grad.abs().sum() > 0
    assert hn.trunk_proj.weight.grad is not None and hn.trunk_proj.weight.grad.abs().sum() > 0
    for p_ in hn._trunk.parameters():
        assert p_.grad is None                        # the trunk itself stays frozen


def test_text_toggles_on_and_off():
    toks = torch.randn(2, 40, D)
    pad = torch.zeros(2, 40, dtype=torch.bool)
    on = _flat(_hn(trunk_text=True), toks, pad, ["open the box", "a"])
    off = _flat(_hn(trunk_text=False), toks, pad, ["open the box", "a"])
    assert not torch.allclose(on, off)


RUN = [test_stride_covers_whole_clip_and_is_even,
       test_output_contract_matches_every_hypernetwork,
       test_left_pad_slots_cannot_reach_the_readout,
       test_gradients_reach_queries_and_proj_but_not_trunk,
       test_text_toggles_on_and_off]
if __name__ == "__main__":
    for fn in RUN:
        fn(); print(f"PASS {fn.__name__}")
    print("ALL TRUNK TESTS PASS")
