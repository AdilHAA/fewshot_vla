"""First-frame bank: one image per (episode, variant) for the vision-conditioning
pairing ablation (init-frame same/cross). Pure numpy/torch."""
from __future__ import annotations

import numpy as np
import torch


def _choice(n: int, g: torch.Generator) -> int:
    return int(torch.randint(n, (1,), generator=g).item())


class FrameBank:
    def __init__(self, path: str):
        z = np.load(path)
        self.images = z["images"]                       # (N,H,W,3) uint8
        self.episode = z["episode"]
        self.task_index = z["task_index"]
        self.variant = (z["variant"] if "variant" in z.files
                        else np.zeros(len(self.episode), dtype=np.int64))
        self._by_ep: dict = {}
        self._by_task: dict = {}
        for i in range(len(self.episode)):
            self._by_ep.setdefault(int(self.episode[i]), []).append(i)
            self._by_task.setdefault(int(self.task_index[i]), []).append(i)

    def _img(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0

    def _pick(self, pool, g: torch.Generator, p_orig: float) -> int:
        """Original frame with prob p_orig (eval conditions on clean env frames, so
        training must see them often), else a random augmented variant."""
        orig = [i for i in pool if int(self.variant[i]) == 0]
        aug = [i for i in pool if int(self.variant[i]) != 0]
        if orig and (not aug or torch.rand(1, generator=g).item() < p_orig):
            return orig[_choice(len(orig), g)]
        return aug[_choice(len(aug), g)]

    def same(self, episode: int, g: torch.Generator, p_orig: float = 0.5) -> torch.Tensor:
        return self._img(self._pick(self._by_ep[int(episode)], g, p_orig))

    def cross(self, task_index: int, exclude_episode: int, g: torch.Generator,
              p_orig: float = 0.5) -> torch.Tensor:
        pool = [i for i in self._by_task.get(int(task_index), [])
                if int(self.episode[i]) != int(exclude_episode)]
        if not pool:
            return self.same(exclude_episode, g, p_orig)
        return self._img(self._pick(pool, g, p_orig))
