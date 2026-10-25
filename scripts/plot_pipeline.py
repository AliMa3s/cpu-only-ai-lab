"""Plot saved whole-pipeline and real document-extraction measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("results")
    comparison = json.loads((root / "pipeline-comparison.json").read_text())
    document = json.loads((root / "document-metrics.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "text.color": "#243447"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.24, top=0.76, wspace=0.36)
    fig.suptitle("CPU-only AI Lab: pipeline and document extraction", x=0.08, y=0.97,
                 ha="left", fontsize=19, weight="bold")
    fig.text(0.08, 0.885, "Ryzen 9 3900X  |  SmolLM2 135M  |  FP32 CPU training and evaluation", fontsize=11)
    blue, green = "#315f91", "#167864"
    costs = [comparison[key]["whole_run"] for key in ("baseline_seconds", "optimized_seconds")]
    axes[0].bar([0, 1], costs, color=[blue, green], width=0.6)
    for x, cost in enumerate(costs):
        axes[0].text(x, cost + max(costs) * 0.025, f"{cost:.1f}s", ha="center")
    axes[0].set(xticks=[0, 1], xticklabels=["Fresh baseline", "Tuned execution"],
                ylabel="Complete run, seconds", ylim=(0, max(costs) * 1.18))
    axes[0].set_title(f"Same SMS training and evaluation work: {costs[0] / costs[1]:.2f}×", loc="left", pad=16)
    axes[0].text(0, -0.29, "120 updates, same examples and validation schedule.\nOne full run per variant; baseline measured first.", transform=axes[0].transAxes, fontsize=9)
    for offset, phase, color, label in ((-0.18, "before", blue, "Before adaptation"),
                                       (0.18, "after", green, "After adaptation")):
        metric = document[phase]["test_generation"]
        values = [100 * metric[key] for key in ("exact_match", "token_f1")]
        axes[1].bar([offset, 1 + offset], values, width=0.32, color=color, label=label)
        for x, value in enumerate(values):
            axes[1].text(x + offset, value + 2, f"{value:.1f}", ha="center", fontsize=10)
    axes[1].set(xticks=[0, 1], xticklabels=["Exact match", "Token F1"], ylabel="Score (%)", ylim=(0, 100))
    axes[1].set_title("Generated answers on real passages", loc="left", pad=16)
    axes[1].legend(frameon=False, loc="upper left", fontsize=9)
    axes[1].text(0, -0.29, "200 filtered SQuAD questions on separate articles.\nShort complete passages; not the full benchmark.", transform=axes[1].transAxes, fontsize=9)
    target = root / "figures"
    target.mkdir(exist_ok=True)
    fig.savefig(target / "pipeline-document.png", dpi=160)
    svg = target / "pipeline-document.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
