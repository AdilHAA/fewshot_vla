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
        self._by_ep: dict = {}
        self._by_task: dict = {}
        for i in range(len(self.episode)):
            self._by_ep.setdefault(int(self.episode[i]), []).append(i)
            self._by_task.setdefault(int(self.task_index[i]), []).append(i)

    def _img(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0

    def same(self, episode: int, g: torch.Generator) -> torch.Tensor:
        pool = self._by_ep[int(episode)]
        return self._img(pool[_choice(len(pool), g)])

    def cross(self, task_index: int, exclude_episode: int, g: torch.Generator) -> torch.Tensor:
        pool = [i for i in self._by_task.get(int(task_index), [])
                if int(self.episode[i]) != int(exclude_episode)]
        if not pool:
            return self.same(exclude_episode, g)
        return self._img(pool[_choice(len(pool), g)])
