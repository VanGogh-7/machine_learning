import gc
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def get_device() -> torch.device:
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed common random number generators for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    """Release unused Python and accelerator memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def save_checkpoint(model: nn.Module, path: str) -> None:
    """Save model parameters to disk."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load model parameters and return the updated model."""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model
