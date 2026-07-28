"""Committed renderer for every result figure the public documents display (audit finding L-4).

Before this script the repository's most public artifact — the headline map-recovery figure — had no
generation path in the tree, no manifest entry and no claim-map coverage, and encoded its text as
glyph paths, so its numbers could not even be read back. Figures are now produced only here, from a
registered result artifact, and ``scripts/validate_artifacts.py`` fails if a displayed result figure
lacks a renderer, a manifest entry or claim-map coverage.

Every figure this script draws is labelled with the estimand it shows. The contrast-correlation
figure is drawn as a **diagnostic**, with the model-free ``singles_zero_prior`` baseline and the
skeleton-alone curve on the same axes, because the raw correlation on its own is confounded by the
purchased lower-order component (audit C-1).

Usage:
    python scripts/render_figures.py --figure map_recovery_diagnostic \
        --recovery report/remediation/corrected_recovery_trpb_johnston2024.json \
        --out figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _console import configure_utf8_stdout

_METHOD_STYLE = {
    "structural": ("#1b6ca8", "o", "structural (coverage order)"),
    "info": ("#c0392b", "s", "info (dispersion-weighted)"),
    "fitness": ("#7f8c8d", "^", "fitness-greedy"),
    "singles_zero_prior": ("#2d8659", "D", "singles + zero prior (model-free)"),
}


def _cell(records: list[dict], method: str, budget: int, order: str, policy: str) -> dict | None:
    for record in records:
        if (
            record.get("method") == method
            and record.get("budget") == budget
            and record.get("order") == order
            and record.get("calibration", {}).get("policy") == policy
        ):
            return record
    return None


def _value(record: dict, field: str) -> float | None:
    if "tie_seed_distribution" in record:
        key = {"raw_pearson_with_skeleton": "raw_pearson_mean"}.get(field)
        return record["tie_seed_distribution"].get(key) if key else None
    return record.get(field)


def render_map_recovery_diagnostic(payload: dict, out: Path) -> Path:
    """Contrast-correlation diagnostic: raw, the skeleton alone, and the model-free baseline."""
    import matplotlib  # noqa: PLC0415  # deferred heavy dependency (figures only)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    records = payload["records"]
    budgets = payload["budgets"]
    policy = "zero_prior"  # decision-eligible, label-free, identical across methods

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, (colour, marker, label) in _METHOD_STYLE.items():
        xs, ys = [], []
        for budget in budgets:
            record = _cell(records, method, budget, "pairwise", policy)
            if record is None:
                continue
            value = _value(record, "raw_pearson_with_skeleton")
            if value is not None:
                xs.append(budget)
                ys.append(value)
        if xs:
            ax.plot(xs, ys, marker=marker, color=colour, label=label, linewidth=1.8)

    skeleton_xs, skeleton_ys = [], []
    for budget in budgets:
        record = _cell(records, "structural", budget, "pairwise", policy)
        if record is None:
            continue
        value = (
            record["tie_seed_distribution"].get("partial_spearman_mean")
            if "tie_seed_distribution" in record
            else record.get("skeleton_pearson")
        )
        if value is not None:
            skeleton_xs.append(budget)
            skeleton_ys.append(value)
    if skeleton_xs:
        ax.plot(
            skeleton_xs,
            skeleton_ys,
            linestyle="--",
            color="#8e44ad",
            marker="x",
            label="structural, skeleton-controlled",
            linewidth=1.6,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("assay budget B (variants)")
    ax.set_ylabel("Pearson r, inferred vs measured pairwise contrast")
    ax.axhline(0.0, color="#bdc3c7", linewidth=0.8)
    ax.set_title(
        f"DIAGNOSTIC — not epistasis-map recovery ({payload['dataset']})",
        fontsize=11,
        color="#8b0000",
    )
    ax.legend(fontsize=8, loc="best", frameon=False)
    fig.text(
        0.5,
        0.5,
        "SUPERSEDED\nsee AUDIT_REMEDIATION_20260728",
        fontsize=26,
        color="#c0392b",
        alpha=0.16,
        ha="center",
        va="center",
        rotation=22,
        transform=ax.transAxes,
        zorder=5,
    )
    fig.text(
        0.01,
        0.01,
        "Raw contrast correlation contains the purchased lower-order component k(S),\n"
        "shared with the truth. The model-free 'singles + zero prior' plate uses no model.\n"
        "Relative SSE gain, not this curve, gates 'recovery' wording.",
        fontsize=6.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"map_recovery_diagnostic_{payload['dataset']}.svg"
    fig.savefig(target, format="svg", metadata={"Date": None})
    plt.close(fig)
    return target


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", required=True, choices=["map_recovery_diagnostic"])
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    payload = json.loads(args.recovery.read_text(encoding="utf-8"))
    if args.figure == "map_recovery_diagnostic":
        target = render_map_recovery_diagnostic(payload, args.out)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
