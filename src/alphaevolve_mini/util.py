"""Small shared helpers: atomic writes and timestamps."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: str | Path, data) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=False) + "\n")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return None
