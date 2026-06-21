from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def checkpoint_path(checkpoint_dir: str, checkpoint_name: str) -> Path:
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / checkpoint_name


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, config_dict: dict) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "config": config_dict,
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, map_location: Optional[str] = None) -> int:
    ckpt = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0))


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 20.0))

