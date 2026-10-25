"""Render saved reader quality, validation constraints and paired CPU latency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def _threshold_order(trial):
    """Place named endpoints around numeric cutoffs, not in search-grid order."""
    threshold = trial["threshold"]
    if threshold == "always_abstain":
        return float("-inf")
    if threshold == "always_answer":
        return float("inf")
    return float(threshold)


def plot(report, selection, benchmark, output: Path, source_hashes=None):
    if report.get("format") != "cpu-reader-report-v1" or selection.get("format") != "cpu-reader-selection-v1":
        raise ValueError("Expected the frozen reader report and threshold selection.")
    choice = selection["choice"]
    if (selection.get("uses_test") is not False or report["selection"].get("test_used_for_selection") is not False
            or any(report["selection"][key] != value for key, value in choice.items())):
        raise ValueError("Figure sources disagree about validation-only deployment.")
    if benchmark is not None:
        if benchmark.get("format") != "cpu-reader-benchmark-v1" or benchmark.get("mode") != "paired":
            raise ValueError("Latency source must be the paired reader benchmark.")
        if benchmark["candidate_checkpoint"] != choice["diagnostic_checkpoint"] or benchmark["candidate_threshold"] != choice["diagnostic_threshold"]:
            raise ValueError("Latency measurement belongs to a different reader candidate.")
        if not all(pair["same_complete_input"] for pair in benchmark["pairs"]):
            raise ValueError("Paired latency requires the same complete question and context.")
    output = Path(output)
    outputs = [output.with_suffix(suffix) for suffix in (".png", ".svg")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Existing reader figures are preserved; choose a new output stem.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#243447", "axes.labelcolor": "#243447", "svg.fonttype": "none"})
    panels = 3 if benchmark is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(20 if panels == 3 else 14, 7))
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.31, top=0.76, wspace=0.32)
    fig.suptitle("CPU-only AI Lab: a compact extractive reader", x=0.05, y=0.97, ha="left", fontsize=22, weight="bold")
    accepted = choice["kind"] == "reader"
    status = f"Selected: {choice['checkpoint']} reader" if accepted else "Reader rejected on validation; generator retained"
    fig.text(0.05, .90, f"135M generator vs 33M reader  |  {status}", fontsize=12)
    fig.text(0.05, .85, "Reader already had SQuAD2 supervision; this CPU run adapts it to fresh DuoRC movie plots.", fontsize=10.5)
    blue, green, amber, gray, red = "#315f91", "#167864", "#b66818", "#7d8792", "#a24444"

    ax = axes[0]
    f1_failures = []
    for name, color, label in (("initial", gray, "Initial reader"), ("final", blue, "After CPU adaptation")):
        trials = sorted((trial for trial in selection["checkpoints"][name]["trials"]
                         if trial["gate"]["exact_correct_retention"] is not None), key=_threshold_order)
        ax.plot([100 * trial["gate"]["exact_correct_retention"] for trial in trials],
                [100 * trial["metrics"]["correct_abstention_recall"] for trial in trials],
                color=color, linewidth=1.6, marker=".", markersize=3, label=label)
        f1_failures.extend(trial for trial in trials if trial["gate"]["checks"]["paired_exact_correct_retention_at_least_0_95"]
                           and trial["gate"]["checks"]["missing_recall_at_least_0_40"]
                           and not trial["gate"]["checks"]["answerable_f1_drop_at_most_0_02"])
    ax.axvline(95, color=green, linestyle="--", linewidth=1)
    ax.axhline(40, color=green, linestyle="--", linewidth=1)
    ax.fill_between([95, 103], 40, 103, color=green, alpha=.08)
    if f1_failures:
        ax.scatter([100 * row["gate"]["exact_correct_retention"] for row in f1_failures],
                   [100 * row["metrics"]["correct_abstention_recall"] for row in f1_failures],
                   marker="x", color=red, s=28, linewidths=1.3, zorder=3, label="Fails the F1 gate")
    diagnostic_selection = selection["checkpoints"][choice["diagnostic_checkpoint"]]
    if accepted:
        diagnostic_metric, diagnostic_gate = diagnostic_selection["selected"], diagnostic_selection["selected_gate"]
    else:
        diagnostic_metric, diagnostic_gate = diagnostic_selection["diagnostic_candidate"]["metrics"], diagnostic_selection["diagnostic_candidate"]["gate"]
    if diagnostic_gate["exact_correct_retention"] is not None:
        ax.scatter([100 * diagnostic_gate["exact_correct_retention"]], [100 * diagnostic_metric["correct_abstention_recall"]],
                   s=150, marker="*", color=amber, edgecolor="white", zorder=4,
                   label=f"Candidate ({choice['diagnostic_checkpoint']})")
    if accepted:
        selected_metric, retention = diagnostic_metric, diagnostic_gate["exact_correct_retention"]
    else:
        selected_metric = selection["checkpoints"]["initial"]["baseline"]
        retention = 1.0 if selected_metric["paired_retention"]["baseline_exact_correct"] else None
    if retention is not None:
        ax.scatter([100 * retention], [100 * selected_metric["correct_abstention_recall"]],
                   s=75, marker="o", facecolor="none", edgecolor="#172f46", linewidth=1.7, zorder=5, label="Selected policy")
    ax.set(xlabel="Generator-correct answers retained (%)", ylabel="Missing questions correctly withheld (%)",
           xlim=(-2, 103), ylim=(-2, 103))
    ax.set_title("Validation chooses the policy", loc="left", pad=18)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.text(0, -.43, "All three gates must pass: retain ≥95% of original correct\nanswers; withhold ≥40% of missing questions; answerable\nF1 may fall by at most 2 points. Test scores do not select.",
            transform=ax.transAxes, fontsize=9, va="top")

    ax = axes[1]
    test = report["test"]
    variants = [test["generator"], test["initial"]["fixed_zero"], test["final"]["fixed_zero"],
                test["diagnostic_candidate"], test["selected"]]
    for offset, key, color, label in ((-.24, "answerable_exact_match", blue, "Answerable EM"),
                                      (0, "answerable_token_f1", green, "Answerable F1"),
                                      (.24, "correct_abstention_recall", amber, "Missing recall")):
        values = [100 * row[key] for row in variants]
        positions = [index + offset for index in range(len(variants))]
        ax.bar(positions, values, width=.22, color=color, label=label)
        for x, value in zip(positions, values, strict=True):
            ax.text(x, value + 2, f"{value:.1f}", ha="center", fontsize=7.4)
    ax.set(xticks=range(len(variants)), xticklabels=["Generator", "Initial\nzero", "Adapted\nzero", "Candidate", "Selected"],
           ylabel="Score (%)", ylim=(0, 117))
    ax.set_title("Held-out quality and adaptation", loc="left", pad=18)
    ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=7.5)
    ax.text(0, -.43, f"{test['generator']['examples']} filtered questions; repeated questions cluster by movie.\nInitial/adapted zero use the same null-score threshold (0).\nCandidate uses its validation-frozen threshold, even if rejected.",
            transform=ax.transAxes, fontsize=9, va="top")

    if benchmark is not None:
        ax = axes[2]
        pairs = benchmark["pairs"]
        ax.set_title("Paired request cost of the candidate", loc="left", pad=18)
        if pairs:
            medians = [1000 * statistics.median(pair[key] for pair in pairs) for key in ("generator_seconds", "reader_seconds")]
            ax.bar([0, 1], medians, width=.6, color=[blue, amber])
            for x, value in enumerate(medians):
                ax.text(x, value + max(medians) * .03, f"{value:.0f} ms", ha="center")
            ax.set(xticks=[0, 1], xticklabels=["Generator", f"Reader ({choice['diagnostic_checkpoint']})"],
                   ylabel="Median warm request latency (ms)", ylim=(0, max(medians) * 1.2))
            ratio = statistics.median(pair["generator_seconds"] / pair["reader_seconds"] for pair in pairs)
            decision = "Selected reader passed the validation gates." if accepted else "Measured reader is rejected; deployed policy stays generator."
            note = (f"{len(pairs)} paired requests; observed median speed ratio {ratio:.2f}×.\n"
                    "Same full inputs; different architectures and outputs.\n"
                    "Tokenization + inference + decoding; loading excluded.\n" + decision)
        else:
            ax.text(.5, .5, "No paired request measurements", ha="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            note = "No paired latency evidence; no speed ratio is claimed."
        ax.text(0, -.43, note, transform=ax.transAxes, fontsize=9, va="top")
    fig.text(.05, .045, "This comparison includes different prior training histories; it does not isolate architecture alone.", fontsize=9, color=gray)
    description = json.dumps({"source_sha256": source_hashes or {}, "selected_kind": choice["kind"],
                              "diagnostic_checkpoint": choice["diagnostic_checkpoint"]}, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[0], dpi=160, metadata={"Description": description})
    fig.savefig(outputs[1], metadata={"Description": description})
    svg = outputs[1]
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/reader/report.json"))
    parser.add_argument("--selection", type=Path, default=Path("results/reader/selection.json"))
    parser.add_argument("--benchmark", type=Path, default=Path("results/reader/benchmark-paired.json"))
    parser.add_argument("--output", type=Path, default=Path("results/figures/reader"))
    args = parser.parse_args()
    report, selection = read(args.report), read(args.selection)
    hashes = {"report.json": sha256(args.report), "selection.json": sha256(args.selection),
              "plot_reader.py": hashlib.sha256(Path(__file__).read_text(encoding="utf-8").encode()).hexdigest()}
    if hashes["selection.json"] != report["artifact_sha256"]["selection.json"]:
        raise ValueError("Selection bytes differ from the completed report binding.")
    benchmark = read(args.benchmark) if args.benchmark.exists() else None
    if benchmark is not None:
        hashes["benchmark-paired.json"] = sha256(args.benchmark)
        if benchmark["source_report_sha256"] != hashes["report.json"]:
            raise ValueError("Benchmark belongs to a different completed reader report.")
    plot(report, selection, benchmark, args.output, hashes)


if __name__ == "__main__":
    main()
