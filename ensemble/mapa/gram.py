"""Linear kernels of train-standardized features, accumulated one column block at a time.

Follows the reproduction's ``mapa_ridge.solve`` (itself MAPA's readout arithmetic): per
feature, mean and population standard deviation of the cell's training rows (zero
deviation becomes one); float32 products; float64 accumulation over column blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Job:
    key: str
    train_pos: np.ndarray          # positions of the training rows in the train features
    target_pos: np.ndarray         # positions of the target rows in the target features
    want_target: bool = False      # also accumulate the target-by-target kernel
    out: dict = field(default_factory=dict)

    def nbytes(self) -> int:
        n, m = len(self.train_pos), len(self.target_pos)
        return 8 * (n * n + m * n + (m * m if self.want_target else 0))


def passes(jobs: list[Job], budget_bytes: int) -> list[list[Job]]:
    """Split jobs so that each pass's accumulators fit the memory budget."""
    groups, current, used = [], [], 0
    for job in jobs:
        if job.nbytes() > budget_bytes:
            raise MemoryError(f"{job.key}: one cell's kernels need {job.nbytes() / 2**30:.1f} GiB, "
                              f"over the {budget_bytes / 2**30:.1f} GiB accumulator budget")
        if current and used + job.nbytes() > budget_bytes:
            groups.append(current)
            current, used = [], 0
        current.append(job)
        used += job.nbytes()
    if current:
        groups.append(current)
    return groups


@torch.inference_mode()
def accumulate(jobs: list[Job], n_columns: int, fetch, device: torch.device, progress=None,
               products: torch.dtype = torch.float32) -> None:
    """Fill ``job.out`` with ``train``, ``cross`` and optionally ``target`` (float64 numpy).

    ``fetch(i)`` returns the i-th aligned column block as two float16 arrays
    ``(train_features[:, col], target_features[:, col])``; the second may be the first.
    ``products`` is the dtype of standardization and products (float32 everywhere except
    the parity check's sensitivity measurement).
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    acc = []
    for job in jobs:
        n, m = len(job.train_pos), len(job.target_pos)
        acc.append({"train": torch.zeros((n, n), dtype=torch.float64, device=device),
                    "cross": torch.zeros((m, n), dtype=torch.float64, device=device),
                    "target": torch.zeros((m, m), dtype=torch.float64, device=device) if job.want_target else None,
                    "tr": torch.as_tensor(job.train_pos, device=device),
                    "tg": torch.as_tensor(job.target_pos, device=device)})
    for index in range(n_columns):
        a_np, b_np = fetch(index)
        a = torch.from_numpy(a_np).to(device)
        b = a if b_np is a_np else torch.from_numpy(b_np).to(device)
        for state in acc:
            x = a[state["tr"]].to(products)
            y = b[state["tg"]].to(products)
            mean = x.mean(0)
            scale = x.std(0, correction=0)
            scale.masked_fill_(scale == 0, 1)
            x.sub_(mean).div_(scale)
            y.sub_(mean).div_(scale)
            state["train"].add_(x @ x.T)
            state["cross"].add_(y @ x.T)
            if state["target"] is not None:
                state["target"].add_(y @ y.T)
        if progress:
            progress(index + 1, n_columns)
    for job, state in zip(jobs, acc):
        job.out = {k: state[k].cpu().numpy() for k in ("train", "cross", "target") if state[k] is not None}
