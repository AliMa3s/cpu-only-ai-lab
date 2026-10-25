"""Render saved validation tradeoffs, held-out quality and optional CPU latency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_bytes())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(report, selection, benchmark, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if report.get("format") != "cpu-document-verifier-report-v1" or selection.get("format") != "cpu-document-verifier-selection-v1":
        raise ValueError("Expected the verifier report and its frozen threshold selection.")
    if selection.get("uses_test") is not False or selection["verifier"]["accepted"] != report["selection"]["accepted"]:
        raise ValueError("Figure sources disagree about validation-only selection.")
    if benchmark is not None and (benchmark.get("format") != "cpu-document-verifier-benchmark-v1" or benchmark.get("mode") != "paired"):
        raise ValueError("Optional latency source must be the paired verifier benchmark.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#243447", "axes.labelcolor": "#243447", "svg.fonttype": "none"})
    panels = 3 if benchmark is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(18 if panels == 3 else 13, 6))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.28, top=0.76, wspace=0.32)
    fig.suptitle("CPU-only AI Lab: learned answer verifier", x=0.055, y=0.97,
                 ha="left", fontsize=21, weight="bold")
    accepted = report["selection"]["accepted"]
    status = "Verifier accepted on validation" if accepted else "Verifier rejected on validation; raw generator retained"
    fig.text(0.055, 0.88, f"Frozen 135M generator  |  21 fitted classifier parameters  |  {status}", fontsize=11)
    blue, green, amber, gray = "#315f91", "#167864", "#b66818", "#7d8792"

    ax = axes[0]
    for name, color, label in (("verifier", blue, "Learned verifier"), ("confidence", gray, "Confidence only")):
        trials = [trial for trial in selection[name]["trials"] if trial["gate"]["exact_correct_retention"] is not None]
        ax.plot([100 * trial["gate"]["exact_correct_retention"] for trial in trials],
                [100 * trial["metrics"]["correct_abstention_recall"] for trial in trials],
                color=color, linewidth=1.6, marker=".", markersize=3, label=label)
    ax.axvline(95, color=green, linestyle="--", linewidth=1)
    ax.axhline(40, color=green, linestyle="--", linewidth=1)
    ax.fill_between([95, 102], 40, 102, color=green, alpha=0.08)
    f1_failures = [trial for name in ("verifier", "confidence") for trial in selection[name]["trials"]
                   if trial["gate"]["checks"]["exact_correct_retention_at_least_0_95"]
                   and trial["gate"]["checks"]["missing_recall_at_least_0_40"]
                   and not trial["gate"]["checks"]["answerable_f1_drop_at_most_0_02"]]
    if f1_failures:
        ax.scatter([100 * trial["gate"]["exact_correct_retention"] for trial in f1_failures],
                   [100 * trial["metrics"]["correct_abstention_recall"] for trial in f1_failures],
                   marker="x", s=28, linewidths=1.3, color="#a24444", zorder=3, label="Fails the F1 gate")
    diagnostic = selection["verifier"]["best_constrained_candidate"]
    if diagnostic is not None:
        ax.scatter([100 * diagnostic["gate"]["exact_correct_retention"]],
                   [100 * diagnostic["metrics"]["correct_abstention_recall"]],
                   s=130, marker="*", color=amber, edgecolor="white", zorder=4, label="Verifier candidate")
    chosen = selection["verifier"]
    retained = chosen["selected_gate"]["exact_correct_retention"] if chosen["accepted"] else 1.0
    ax.scatter([100 * retained], [100 * chosen["selected"]["correct_abstention_recall"]],
               s=70, marker="o", facecolor="none", edgecolor="#172f46", linewidth=1.6,
               zorder=5, label="Selected policy")
    ax.set(xlabel="Baseline exact-correct answers retained (%)", ylabel="Missing questions correctly withheld (%)",
           xlim=(-2, 103), ylim=(-2, 103))
    ax.set_title("Validation chooses the policy", loc="left", pad=18)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.text(0, -0.37, "Dashed gates: retain ≥95%; withhold ≥40% of missing.\nAnswerable F1 may drop by at most 2 points (also required).\nThe star is diagnostic if no threshold passes all gates.",
            transform=ax.transAxes, fontsize=8.6, va="top")

    ax = axes[1]
    variants = ("baseline", "verifier_candidate", "selected")
    for offset, key, color, label in ((-0.24, "answerable_exact_match", blue, "Answerable EM"),
                                      (0, "answerable_token_f1", green, "Answerable F1"),
                                      (0.24, "correct_abstention_recall", amber, "Missing recall")):
        values = [100 * report["test"][name][key] for name in variants]
        positions = [index + offset for index in range(len(variants))]
        ax.bar(positions, values, width=0.22, color=color, label=label)
        for x, value in zip(positions, values):
            ax.text(x, value + 2, f"{value:.1f}", ha="center", fontsize=8)
    ax.set(xticks=range(3), xticklabels=["Raw generator", "Verifier\ncandidate", "Selected"], ylabel="Score (%)", ylim=(0, 115))
    ax.set_title("Held-out quality", loc="left", pad=18)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=7.6)
    examples = report["test"]["baseline"]["examples"]
    ax.text(0, -0.37, f"{examples} new filtered SQuAD2 questions; balanced four strata.\nCandidate is fixed using validation, even when rejected.\nHeld-out results never change the threshold or decision.",
            transform=ax.transAxes, fontsize=8.6, va="top")

    if benchmark is not None:
        ax = axes[2]
        matched = [pair for pair in benchmark["pairs"] if pair["matched_work"]]
        ax.set_title("Cost of running the candidate", loc="left", pad=18)
        if matched:
            medians = [1000 * statistics.median(pair[field] for pair in matched)
                       for field in ("baseline_seconds", "verifier_seconds")]
            ax.bar([0, 1], medians, width=0.6, color=[blue, amber])
            for x, value in enumerate(medians):
                ax.text(x, value + max(medians) * 0.025, f"{value:.0f} ms", ha="center")
            ax.set(xticks=[0, 1], xticklabels=["Raw generator", "With verifier"],
                   ylabel="Median warm request latency (ms)", ylim=(0, max(medians) * 1.2))
            ratio = statistics.median(pair["verifier_seconds"] / pair["baseline_seconds"] for pair in matched)
            note = f"{len(matched)}/{len(benchmark['pairs'])} pairs have matching generated work.\nMedian paired latency ratio: {ratio:.3f}×.\nThis small timing sample does not establish a speedup.\nCandidate cost is measured even when deployment rejects it."
        else:
            ax.text(0.5, 0.5, "No matched-work latency pairs", ha="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            note = "The benchmark supplied no matched-work pairs.\nNo latency ratio is claimed."
        ax.text(0, -0.37, note, transform=ax.transAxes, fontsize=8.6, va="top")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=160)
    svg = output.with_suffix(".svg")
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/document-verifier/report.json"))
    parser.add_argument("--selection", type=Path, default=Path("results/document-verifier/selection.json"))
    parser.add_argument("--benchmark", type=Path, default=Path("results/document-verifier/benchmark-paired.json"))
    parser.add_argument("--output", type=Path, default=Path("results/figures/document-verifier"))
    args = parser.parse_args()
    report, selection = read(args.report), read(args.selection)
    if sha256(args.selection) != report["artifact_sha256"]["selection.json"]:
        raise ValueError("Selection bytes do not match the completed report.")
    benchmark = read(args.benchmark) if args.benchmark.exists() else None
    if benchmark is not None and benchmark["source_report_sha256"] != sha256(args.report):
        raise ValueError("Latency benchmark belongs to a different completed report.")
    plot(report, selection, benchmark, args.output)


if __name__ == "__main__":
    main()
