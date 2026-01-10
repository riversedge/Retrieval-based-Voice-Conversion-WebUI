import os
from typing import Optional

import torch


_VALID_PREFER = {"auto", "cpu", "mps", "cuda"}


def _normalize_prefer(prefer: Optional[str]) -> str:
    if prefer is None:
        return "auto"
    prefer = str(prefer).strip().lower()
    return prefer if prefer else "auto"


def is_cuda_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def is_mps_available() -> bool:
    if not getattr(torch.backends, "mps", None):
        return False
    if not torch.backends.mps.is_available():
        return False
    try:
        torch.zeros(1).to(torch.device("mps"))
        return True
    except Exception:
        return False


def get_device(prefer: str = "auto") -> torch.device:
    env_prefer = os.getenv("RVC_DEVICE")
    if env_prefer:
        prefer = env_prefer
    prefer = _normalize_prefer(prefer)
    if prefer.startswith("cuda"):
        if is_cuda_available():
            return torch.device("cuda:0" if prefer in {"cuda", "cuda:0"} else prefer)
        return torch.device("cpu")
    if prefer == "mps":
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer not in _VALID_PREFER and ":" in prefer:
        # Allow explicit device strings like cuda:1 if available.
        if "cuda" in prefer and is_cuda_available():
            return torch.device(prefer)
        if "mps" in prefer and is_mps_available():
            return torch.device("mps")
    if is_cuda_available():
        return torch.device("cuda:0")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_str(prefer: str = "auto") -> str:
    if isinstance(prefer, torch.device):
        return str(prefer)
    if isinstance(prefer, str) and prefer.strip().lower().startswith("xpu"):
        return prefer
    return str(get_device(prefer))
