"""Backward-compatible alias for the canonical validity-audit runner."""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("run_validity_audit.py")),
        run_name="__main__",
    )
