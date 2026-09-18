"""Render the current CERT-FLOW upgrade dashboard from full-run artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets"
FLEET = ROOT / "results/v4_validity/fleet-sensing-full.json"
RECOVERY = ROOT / "results/v4_validity/shift-recovery-full.json"
REALWORLD = ROOT / "results/v4_validity/realworld-audit-full.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def theme() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.facecolor": "#101820",
        "figure.facecolor": "#0b1117",
        "axes.edgecolor": "#425466",
        "axes.labelcolor": "#dce7f2",
        "xtick.color": "#b8c7d6",
        "ytick.color": "#b8c7d6",
        "text.color": "#e7eef6",
        "grid.color": "#2b3b4b",
        "grid.alpha": 0.65,
    })


def render_dashboard() -> Path:
    fleet = load(FLEET)
    recovery = load(RECOVERY)
    realworld = load(REALWORLD)
    theme()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(
        "CERT-FLOW current system | certified planning, sensing, recovery",
        fontsize=18, fontweight="bold", color="#ffffff",
    )

    ax = axes[0, 0]
    budgets = sorted({int(row["budget"]) for row in fleet["rows"]
                      if str(row["budget"]) != "inf"})
    policies = ["cert", "hybrid", "voi", "random", "max_age", "max_width"]
    colors = {"cert": "#56b4e9", "hybrid": "#009e73", "voi": "#e69f00",
              "random": "#cc79a7", "max_age": "#d55e00", "max_width": "#999999"}
    for policy in policies:
        values = []
        for budget in budgets:
            rows = [r for r in fleet["rows"]
                    if r["policy"] == policy and int(r["budget"]) == budget]
            values.append(rows[0]["regret_median"] if rows else np.nan)
        ax.plot(budgets, values, marker="o", linewidth=2.5, label=policy,
                color=colors[policy])
    ax.set_title("Budgeted sensing: median route regret ↓", loc="left", fontweight="bold")
    ax.set_xlabel("sensing budget")
    ax.set_ylabel("regret")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[0, 1]
    methods = ["aci", "heuristic", "formal"]
    labels = ["ACI", "heuristic", "formal SR"]
    values = [recovery["aggregate"][m]["post_edge_coverage_mean"] for m in methods]
    bars = ax.bar(labels, values, color=["#999999", "#e69f00", "#009e73"], width=0.62)
    ax.axhline(0.90, color="#d55e00", linestyle="--", linewidth=1.5,
               label="visual reference (not a gate)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006,
                f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0.88, 0.99)
    ax.set_title("Abrupt-shift edge coverage ↑", loc="left", fontweight="bold")
    ax.set_ylabel("held-out coverage")
    ax.grid(True, axis="y")
    detection = recovery["aggregate"]["formal"].get("detection_delay_median", np.nan)
    recovery_delay = recovery["aggregate"]["formal"].get("recovery_delay_median", np.nan)
    reentry = recovery["aggregate"]["formal"].get("recovery_to_valid_median", np.nan)
    annotation = (
        "formal: "
        f"{detection:.0f} detection · {recovery_delay:.0f} recovery · "
        f"{reentry:.0f} re-entry"
        if all(np.isfinite(v) for v in (detection, recovery_delay, reentry))
        else "formal timing unavailable in artifact"
    )
    ax.text(0.02, 0.03, annotation,
            transform=ax.transAxes, fontsize=8, color="#b8c7d6")

    ax = axes[1, 0]
    datasets = ["METR-LA", "PEMS-BAY"]
    selection = [realworld["aggregate"][k]["selection_audit_lcb_min"]
                 for k in ("metr-la", "pems-bay")]
    group = [realworld["aggregate"][k]["group_min_lcb_min"]
             for k in ("metr-la", "pems-bay")]
    x = np.arange(len(datasets))
    width = 0.34
    ax.bar(x - width / 2, selection, width, label="post-selection audit",
           color="#56b4e9")
    ax.bar(x + width / 2, group, width, label="finite-group audit",
           color="#009e73")
    ax.axhline(0.80, color="#d55e00", linestyle="--", linewidth=1.5,
               label="release floor")
    ax.set_xticks(x, datasets)
    ax.set_ylim(0.75, 0.90)
    ax.set_ylabel("lower confidence bound")
    ax.set_title("Fresh conditional audits ↑", loc="left", fontweight="bold")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    formal = recovery["aggregate"]["formal"]
    formal_false_alarm = formal.get("pre_false_alarm_fraction", float("nan"))
    formal_recovery = formal.get("recovery_to_valid_median", float("nan"))
    path_coverage = formal.get("path_coverage_mean", float("nan"))
    realworld_lcb = min(
        realworld["aggregate"][key]["selection_audit_lcb_min"]
        for key in ("metr-la", "pems-bay")
    )
    facts = [
        (f"{recovery.get('seeds', '?')} seeds", "shift-recovery artifact"),
        (f"{path_coverage:.3f}" if np.isfinite(path_coverage) else "n/a",
         "formal path coverage"),
        (f"{formal_false_alarm:.3f}" if np.isfinite(formal_false_alarm) else "n/a",
         "formal pre-shift false-alarm fraction"),
        (f"{formal_recovery:.0f} rounds" if np.isfinite(formal_recovery) else "n/a",
         "fresh re-entry median"),
        (f"{realworld_lcb:.3f}" if np.isfinite(realworld_lcb) else "n/a",
         "minimum real-world selection LCB"),
        ("bound", "values read from persisted artifacts"),
    ]
    ax.text(0.02, 0.96, "Release evidence", fontsize=14, fontweight="bold",
            va="top", color="#ffffff")
    for i, (value, label) in enumerate(facts):
        y = 0.80 - i * 0.135
        ax.text(0.04, y, value, fontsize=18, fontweight="bold", color="#56b4e9")
        ax.text(0.34, y + 0.005, label, fontsize=10, color="#dce7f2")
    ax.text(0.02, 0.04, "Full-run artifacts: results/v4_validity/", fontsize=8,
            color="#91a4b8")

    target = OUT / "current_system_dashboard.png"
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target


def render_recovery_frontier() -> Path:
    recovery = load(RECOVERY)
    theme()
    fig, ax = plt.subplots(figsize=(9, 5))
    formal = recovery["aggregate"]["formal"]
    aci = recovery["aggregate"]["aci"]
    heuristic = recovery["aggregate"]["heuristic"]
    names = ["ACI", "heuristic", "formal SR"]
    coverage = [aci["post_edge_coverage_mean"], heuristic["post_edge_coverage_mean"],
                formal["post_edge_coverage_mean"]]
    bars = ax.bar(names, coverage, color=["#999999", "#e69f00", "#009e73"])
    ax.axhline(0.90, color="#d55e00", linestyle="--",
               label="visual reference (not a gate)")
    for bar, value in zip(bars, coverage):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002,
                f"{value:.3f}", ha="center")
    ax.set_ylim(0.90, 0.98)
    ax.set_ylabel("post-shift held-out edge coverage")
    ax.set_title("Recovery frontier | formal monitoring raises post-shift coverage")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    target = OUT / "recovery_frontier.png"
    fig.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return target


if __name__ == "__main__":
    print(render_dashboard())
    print(render_recovery_frontier())
