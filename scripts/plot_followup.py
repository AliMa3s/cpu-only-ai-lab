"""Render the follow-up measurements; optional dependency: matplotlib==3.10.7."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    bench = read(args.results / "extended-benchmark.json")
    small = read(args.results / "real-task-135m.json")
    larger = read(args.results / "real-task-360m.json")
    scratch = read(args.results / "scratch.json")
    curve = read(args.results / "scratch-learning-curve.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#243447",
                         "text.color": "#243447", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.24, top=0.79, wspace=0.34)
    fig.suptitle("CPU-only AI Lab: follow-up measurements", x=0.055, ha="left", y=0.96, fontsize=21, weight="bold")
    fig.text(0.055, 0.875, "AMD Ryzen 9 3900X  |  CPU training and inference  |  All figures come from the saved local runs", fontsize=11)
    blue, green, gray = "#315f91", "#167864", "#a1aab5"

    ax = axes[0]
    summaries = bench["summary"]["by_token_budget"]
    budgets = sorted(summaries, key=int)
    for i, key in enumerate(budgets):
        row = summaries[key]
        if not row["pairs"]:
            ax.text(i, 1, "No matched pairs", ha="center")
            continue
        for offset, field, color, label in (
            (-0.19, "pytorch_aggregate_tokens_per_second", blue, "Tuned PyTorch FP32"),
            (0.19, "ollama_aggregate_tokens_per_second", green, "Ollama F16"),
        ):
            value = row[field]
            ax.bar(i + offset, value, width=0.34, color=color, label=label if i == 0 else None)
            ax.text(i + offset, value + 1.2, f"{value:.1f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(budgets)), [f"{b} output tokens" for b in budgets])
    ax.set_title("Longer generation", loc="left", pad=14)
    ax.set_ylabel("Generated tokens / second (wall time)")
    ax.set_ylim(bottom=0, top=max([r.get("ollama_aggregate_tokens_per_second", 1) for r in summaries.values()]) * 1.25)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(0, -0.30, "Exact prompt/output token counts required.\nRuntime and weight precision both change.", transform=ax.transAxes, fontsize=9)

    ax = axes[1]
    labels = ["135M\nbefore", "135M\nafter", "360M\nbefore", "360M\nafter", "Naive\nBayes"]
    values = [100 * m[stage]["test"]["macro_f1"] for m in (small, larger) for stage in ("before", "after")]
    values.append(100 * small["classical_baselines"]["naive_bayes"]["test"]["macro_f1"])
    bars = ax.bar(range(len(values)), values, color=[gray, green, gray, green, blue], width=0.68)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + 1.2, f"{value:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_title("Real SMS classification", loc="left", pad=14)
    ax.set_ylabel("Macro F1 (%) on 512 held-out messages")
    ax.text(0, -0.30, "Constrained binary classification, same test set.\nNB uses all training rows; LoRA has a step budget.", transform=ax.transAxes, fontsize=9)

    ax = axes[2]
    validation = curve["validation"]
    ax.plot([r["step"] for r in validation], [r["loss_nats_per_byte"] for r in validation], color=green, marker="o", linewidth=2, label="Validation")
    ax.axhline(scratch["before"]["test_unigram"]["loss_nats_per_byte"], color=gray, linestyle="--", label="Unigram test baseline")
    ax.scatter([scratch["selection"]["best_step"]], [scratch["after"]["test"]["loss_nats_per_byte"]], color=blue, marker="D", zorder=5, label="Final held-out test")
    ax.set_title("Training from random weights", loc="left", pad=14)
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Next-byte loss (nats; lower is better)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=9)
    ax.text(0, -0.30, "1.90M parameters; contiguous Shakespeare splits.\nByte loss is not a measure of general chat ability.", transform=ax.transAxes, fontsize=9)

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e6eaef", linewidth=0.7)
    destination = args.results / "figures"
    destination.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(destination / f"followup.{extension}", dpi=150, facecolor="white", metadata={"Creator": "CPU-only AI Lab"} if extension == "svg" else None)
        if extension == "svg":
            svg = destination / "followup.svg"
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print("Wrote results/figures/followup.png and followup.svg")


if __name__ == "__main__":
    main()
