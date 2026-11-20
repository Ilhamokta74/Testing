from __future__ import annotations

import gc
import threading
from contextlib import contextmanager

from .util import ClipperError


def free_vram() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class GpuManager:
    """Guarantees only ONE GPU model is active at a time (6 GB VRAM)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active: str | None = None

    def acquire(self, name: str) -> None:
        with self._lock:
            if self.active:
                raise ClipperError(f"GPU still held by '{self.active}', cannot load '{name}'.")
            self.active = name

    def release(self) -> None:
        with self._lock:
            self.active = None
        free_vram()

    @contextmanager
    def slot(self, name: str):
        self.acquire(name)
        try:
            yield
        finally:
            self.release()


GPU = GpuManager()


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda failed" in msg or "cublas" in msg
