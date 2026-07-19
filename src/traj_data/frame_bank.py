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
        return self.cross_set(task_index, exclude_episode, g, 1, p_orig)[0]

    def cross_set(self, task_index: int, exclude_episode: int, g: torch.Generator,
                  k: int, p_orig: float = 0.5) -> list:
        """k t=0 frames from k DISTINCT other episodes of the task (repeats only when
        fewer exist; single-episode tasks fall back to the episode itself)."""
        eps = sorted({int(self.episode[i])
                      for i in self._by_task.get(int(task_index), [])
                      if int(self.episode[i]) != int(exclude_episode)})
        if not eps:
            return [self.same(exclude_episode, g, p_orig) for _ in range(k)]
        if len(eps) >= k:
            perm = torch.randperm(len(eps), generator=g).tolist()
            chosen = [eps[i] for i in perm[:k]]
        else:
            chosen = [eps[_choice(len(eps), g)] for _ in range(k)]
        return [self._img(self._pick(self._by_ep[e], g, p_orig)) for e in chosen]
