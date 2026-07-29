from __future__ import annotations

from typing import Literal

DeviceName = Literal["cpu", "cuda"]


def resolve_device(device: str = "auto") -> DeviceName:
    """Resolve auto|cpu|cuda to a concrete device string."""
    choice = device.lower().strip()
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not _cuda_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    if choice == "auto":
        return "cuda" if _cuda_available() else "cpu"
    raise ValueError(f"Unsupported device: {device!r}")


def default_compute_type(device: DeviceName, compute_type: str | None = None) -> str:
    if compute_type:
        return compute_type
    return "float16" if device == "cuda" else "int8"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
