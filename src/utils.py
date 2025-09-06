import os
import random
import time
from datetime import datetime, date
import numpy as np
import torch


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def today_str_local() -> str:
    # Simple local date; good enough for daily cache invalidation
    return date.today().isoformat()


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
