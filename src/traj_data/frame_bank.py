"""First-frame bank: the t=0 frame of every ORIGINAL episode, for the vision
init-frame pairing (p_self). Sim-augmented variants in the bank file are ignored.
Pure numpy/torch."""
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
        variant = (z["variant"] if "variant" in z.files
                   else np.zeros(len(self.episode), dtype=np.int64))
        self._by_ep: dict = {}
        self._by_task: dict = {}
        for i in range(len(self.episode)):
            if int(variant[i]) != 0:                    # originals only
                continue
            self._by_ep.setdefault(int(self.episode[i]), []).append(i)
            self._by_task.setdefault(int(self.task_index[i]), []).append(i)

    def _img(self, i: int) -> torch.Tensor:
        return torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0

    def same(self, episode: int, g: torch.Generator) -> torch.Tensor:
        pool = self._by_ep[int(episode)]
        return self._img(pool[_choice(len(pool), g)])

    def cross(self, task_index: int, exclude_episode: int, g: torch.Generator) -> torch.Tensor:
        """t=0 frame of one random OTHER episode of the task; single-episode tasks
        fall back to the episode itself."""
        pool = [i for i in self._by_task.get(int(task_index), [])
                if int(self.episode[i]) != int(exclude_episode)]
        if not pool:
            return self.same(exclude_episode, g)
        return self._img(pool[_choice(len(pool), g)])
