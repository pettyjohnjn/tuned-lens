from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # no runtime imports
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LambdaLR
    from torchdata.dataloader2 import DataLoader2
    from tuned_lens.nn.lenses import Lens  # type only

@dataclass
class State:
    """All of the stateful information in the training loop."""
    dataloader: "DataLoader2"
    lens: "Lens"
    opt: "Optimizer"
    scheduler: "LambdaLR"
    wandb_id: Optional[str]
    nats_to_bpb: float
    step: int = 0

    def load(self, snapshot_file: Path, device) -> None:
        """Load a snapshot file."""
        logger.info(f"Loading snapshot from {snapshot_file}...")
        import torch as th  # lazy
        snapshot = th.load(snapshot_file, map_location=device)
        self.step = snapshot["step"]
        self.wandb_id = snapshot["wandb_id"]
        self.lens.load_state_dict(snapshot["lens"])
        self.opt.load_state_dict(snapshot["optim"])
        self.scheduler.load_state_dict(snapshot["scheduler"])
        self.dataloader.load_state_dict(snapshot["dataloader"])

    def save(self, snapshot_file: Path) -> None:
        """Save a snapshot file."""
        logger.info(f"Saving snapshot to {snapshot_file}...")
        import torch as th  # lazy

        # ZeroRedundancyOptimizer import guarded to avoid cold-start cost
        try:
            from torch.distributed.optim import ZeroRedundancyOptimizer  # type: ignore
            is_zro = isinstance(self.opt, ZeroRedundancyOptimizer)
        except Exception:
            is_zro = False

        if is_zro:
            # type: ignore[attr-defined]
            self.opt.consolidate_state_dict()

        th.save(
            {
                "lens": self.lens.state_dict(),
                "optim": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "dataloader": self.dataloader.state_dict(),
                "step": self.step,
                "wandb_id": self.wandb_id,
            },
            snapshot_file,
        )

__all__ = ["State"]