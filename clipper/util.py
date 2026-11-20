from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


class ClipperError(Exception):
    """Error with a clear, user-facing message."""


def slugify(text: str, maxlen: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return (text or "video")[:maxlen].strip("-") or "video"


def run(cmd: list[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [str(c) for c in cmd], cwd=cwd, check=True, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as e:
        raise ClipperError(f"Command not found: {cmd[0]}. Install it and add it to PATH.") from e
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip()[-1500:]
        raise ClipperError(f"{cmd[0]} failed:\n{tail}") from e


def hash_key(*parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def file_hash(path: str | Path) -> str:
    """Fast fingerprint: size + first/last 4 MB."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with p.open("rb") as f:
        h.update(f.read(4 << 20))
        if size > (8 << 20):
            f.seek(-(4 << 20), 2)
            h.update(f.read(4 << 20))
    return h.hexdigest()[:16]


def fmt_ts(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
