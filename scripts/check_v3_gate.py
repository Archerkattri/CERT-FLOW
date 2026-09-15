"""Backward-compatible alias for :mod:`check_release_gate`.

The release checker has one implementation. This historical filename remains
usable for old links without creating a second gate that can drift.
"""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("check_release_gate.py")),
        run_name="__main__",
    )
