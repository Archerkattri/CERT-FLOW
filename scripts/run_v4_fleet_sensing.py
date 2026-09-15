"""Backward-compatible alias for the canonical fleet-sensing runner."""
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("run_fleet_sensing.py")),
        run_name="__main__",
    )
