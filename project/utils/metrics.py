from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class RunningMetrics:
    n: int = 0
    mae_sum: float = 0.0
    mse_sum: float = 0.0
    gt_sum: float = 0.0
    gt_sq_sum: float = 0.0

    def update(self, pred_count: torch.Tensor, gt_count: torch.Tensor) -> None:
        diff = pred_count.detach() - gt_count.detach()
        gt = gt_count.detach()
        self.mae_sum += torch.abs(diff).sum().item()
        self.mse_sum += torch.square(diff).sum().item()
        self.gt_sum += gt.sum().item()
        self.gt_sq_sum += torch.square(gt).sum().item()
        self.n += int(diff.numel())

    @property
    def mae(self) -> float:
        return self.mae_sum / max(1, self.n)

    @property
    def rmse(self) -> float:
        return math.sqrt(self.mse_sum / max(1, self.n))

    @property
    def r2(self) -> float:
        if self.n <= 1:
            return float("nan")
        ss_tot = self.gt_sq_sum - (self.gt_sum * self.gt_sum) / self.n
        if ss_tot <= 1e-12:
            return float("nan")
        return 1.0 - (self.mse_sum / ss_tot)
