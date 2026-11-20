from __future__ import annotations

import shutil
from pathlib import Path

from .util import read_json, write_json


class StageCache:
    """Plain-JSON stage cache (no database). Keys live in work/<name>/_keys.json."""

    def __init__(self, work_dir: Path, enabled: bool = True, fresh: bool = False):
        self.dir = Path(work_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.keys_path = self.dir / "_keys.json"
        self.keys: dict = {}
        if self.keys_path.exists() and not fresh:
            try:
                self.keys = read_json(self.keys_path)
            except Exception:
                self.keys = {}

    def path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def get(self, name: str, key: str):
        p = self.path(name)
        if self.enabled and p.exists() and self.keys.get(name) == key:
            try:
                return read_json(p)
            except Exception:
                return None
        return None

    def put(self, name: str, key: str, data) -> None:
        write_json(self.path(name), data)
        self.keys[name] = key
        write_json(self.keys_path, self.keys)

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
