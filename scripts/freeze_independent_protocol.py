"""Freeze the outcome-blind day/map evaluation manifest for CERT-FLOW.

The script reads file identity and timestamp boundaries only. It never runs a
planner or computes a metric on the locked endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "protocols" / "independent-day-map-v1.json"

TRAFFIC_SPLITS = {
    "metr-la": {
        "path": "data/metr-la/metr_la.h5",
        "development": ("2012-03-01T00:00:00", "2012-05-31T00:00:00"),
        "validation": ("2012-05-31T00:00:00", "2012-06-14T00:00:00"),
        "locked_test": ("2012-06-14T00:00:00", "2012-06-28T00:00:00"),
    },
    "pems-bay": {
        "path": "data/pems-bay/pems_bay.h5",
        "development": ("2017-01-01T00:00:00", "2017-05-01T00:00:00"),
        "validation": ("2017-05-01T00:00:00", "2017-06-01T00:00:00"),
        "locked_test": ("2017-06-01T00:00:00", "2017-07-01T00:00:00"),
    },
}

MAP_SPLITS = {
    "development": ("data/movingai/dao/arena.map", "data/movingai/dao/arena.map.scen"),
    "validation": ("data/movingai/Berlin_2_512.map", "data/movingai/Berlin_2_512.map.scen"),
    "locked_test": ("data/movingai/Boston_2_1024.map", "data/movingai/Boston_2_1024.map.scen"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _traffic_entry(name: str, spec: dict[str, object]) -> dict[str, object]:
    relative = str(spec["path"])
    path = ROOT / relative
    frame = pd.read_hdf(path)
    first = frame.index[0].to_pydatetime()
    last_exclusive = frame.index[-1].to_pydatetime() + (frame.index[1] - frame.index[0])
    windows = {}
    previous_stop: datetime | None = None
    for role in ("development", "validation", "locked_test"):
        start_text, stop_text = spec[role]  # type: ignore[misc]
        start = datetime.fromisoformat(start_text)
        stop = datetime.fromisoformat(stop_text)
        if not first <= start < stop <= last_exclusive:
            raise ValueError(f"{name} {role} falls outside the recording")
        if previous_stop is not None and start != previous_stop:
            raise ValueError(f"{name} windows must be contiguous and disjoint")
        previous_stop = stop
        windows[role] = {
            "start_inclusive": start.isoformat(),
            "stop_exclusive": stop.isoformat(),
            "bins": int((stop - start).total_seconds() // 300),
            "metric_status": "LOCKED_UNRUN" if role == "locked_test" else "AVAILABLE",
        }
    return {
        "dataset": name,
        "path": relative,
        "sha256": _sha256(path),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "recording_start": first.isoformat(),
        "recording_stop_exclusive": last_exclusive.isoformat(),
        "windows": windows,
    }


def _map_entry(role: str, paths: tuple[str, str]) -> dict[str, object]:
    map_path, scenario_path = (ROOT / value for value in paths)
    if not map_path.is_file() or not scenario_path.is_file():
        raise FileNotFoundError(f"missing MovingAI {role} map or scenario")
    return {
        "role": role,
        "map": paths[0],
        "map_sha256": _sha256(map_path),
        "scenario": paths[1],
        "scenario_sha256": _sha256(scenario_path),
        "metric_status": "LOCKED_UNRUN" if role == "locked_test" else "AVAILABLE",
    }


def build_manifest() -> dict[str, object]:
    return {
        "schema": "certflow.independent-day-map.v1",
        "frozen_on": "2026-09-15",
        "outcome_blind": True,
        "selection_rule": {
            "traffic_endpoints": "lexicographically first reachable pair with at least six hops",
            "movingai_scenarios": "first 100 valid scenario rows in file order",
            "seeds": list(range(10)),
        },
        "use_policy": {
            "development": "method and hyperparameter changes allowed",
            "validation": "one configuration selection after development",
            "locked_test": "single final evaluation after code/config hash freeze; no retuning",
        },
        "traffic": [
            _traffic_entry(name, spec) for name, spec in TRAFFIC_SPLITS.items()
        ],
        "movingai": [
            _map_entry(role, paths) for role, paths in MAP_SPLITS.items()
        ],
        "release_gate": {
            "requires_clean_source_tree": True,
            "requires_manifest_hash_in_report": True,
            "requires_all_failures_and_abstentions": True,
            "locked_endpoint_metrics_present": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("frozen protocol manifest is missing or stale")
        print(f"verified {OUTPUT}")
        return 0
    if OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") != rendered:
        raise SystemExit("refusing to overwrite a different frozen protocol manifest")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
