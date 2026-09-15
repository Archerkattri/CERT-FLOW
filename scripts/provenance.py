"""Small, deterministic provenance helpers for CERT-FLOW experiments."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable


def _source_files(root: Path, roots: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for relative in roots:
        base = root / relative
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(root)
            if "__pycache__" in relative_path.parts:
                continue
            if relative_path.parts[:2] == ("scripts", "out"):
                continue
            if path.suffix.lower() in {".py", ".toml", ".yaml", ".yml", ".json"}:
                paths.append(path)
    return sorted(set(paths))


def source_manifest(root: Path, roots: Iterable[str] = ("src", "scripts")) -> dict:
    """Return a content identity for executable/configuration source.

    A Git revision alone is insufficient while the repository is dirty.  The
    digest includes tracked and untracked source files plus the complete Git
    status text, while excluding generated result/checkpoint directories.
    """
    digest = hashlib.sha256()
    files = _source_files(root, roots)
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(relative + b"\0" + content + b"\0")
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        revision, status = "unknown", "unknown"
    return {
        "revision": revision,
        "source_digest": digest.hexdigest(),
        "source_file_count": len(files),
        "working_tree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "dirty_tree": status not in ("", "unknown"),
    }
