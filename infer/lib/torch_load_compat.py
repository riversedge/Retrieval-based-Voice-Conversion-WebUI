import inspect
from contextlib import contextmanager, nullcontext
from typing import Any, Iterable, Optional

import torch


def _supports_param(param: str) -> bool:
    try:
        return param in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False


def _safe_globals_context(safe_globals: Optional[Iterable[Any]]):
    if not safe_globals:
        return nullcontext()
    safe_globals = list(safe_globals)
    if hasattr(torch.serialization, "safe_globals"):
        return torch.serialization.safe_globals(safe_globals)
    return nullcontext()


def torch_load_compat(
    path,
    map_location=None,
    weights_only_default: bool = False,
    safe_globals: Optional[Iterable[Any]] = None,
):
    """Compatibility wrapper for torch.load across 2.x releases."""
    kwargs = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    if _supports_param("weights_only"):
        kwargs["weights_only"] = weights_only_default
    if safe_globals and _supports_param("safe_globals"):
        kwargs["safe_globals"] = list(safe_globals)

    with _safe_globals_context(safe_globals):
        try:
            return torch.load(path, **kwargs)
        except Exception:
            if _supports_param("weights_only") and kwargs.get("weights_only"):
                kwargs["weights_only"] = False
                return torch.load(path, **kwargs)
            raise
