from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "protocols" / "independent-day-map-v1.json"


def test_independent_protocol_is_disjoint_and_locked() -> None:
    protocol = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert protocol["schema"] == "certflow.independent-day-map.v1"
    assert protocol["outcome_blind"] is True
    assert protocol["release_gate"]["locked_endpoint_metrics_present"] is False

    for dataset in protocol["traffic"]:
        windows = dataset["windows"]
        development_stop = datetime.fromisoformat(windows["development"]["stop_exclusive"])
        validation_start = datetime.fromisoformat(windows["validation"]["start_inclusive"])
        validation_stop = datetime.fromisoformat(windows["validation"]["stop_exclusive"])
        test_start = datetime.fromisoformat(windows["locked_test"]["start_inclusive"])
        assert development_stop == validation_start
        assert validation_stop == test_start
        assert windows["locked_test"]["metric_status"] == "LOCKED_UNRUN"

    maps = {entry["role"]: entry for entry in protocol["movingai"]}
    assert set(maps) == {"development", "validation", "locked_test"}
    assert maps["locked_test"]["metric_status"] == "LOCKED_UNRUN"
    assert len({entry["map_sha256"] for entry in maps.values()}) == 3
