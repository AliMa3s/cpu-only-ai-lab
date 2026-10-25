"""Render measured answer accuracy and abstention by passage length."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path("results")
    metrics = json.loads((root / "document-v2-metrics.json").read_text(encoding="utf-8"))
    trials = json.loads((root / "document-v2-selection-trials.json").read_text(encoding="utf-8"))
    guard = json.loads((root / "document-v2-guard/report.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "text.color": "#243447"})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.26, top=0.75, wspace=0.32)
    fig.suptitle("CPU-only AI Lab: longer passages and missing answers", x=0.055, y=0.965, ha="left", fontsize=22, weight="bold")
    fig.text(0.055, 0.875, "SmolLM2 135M  |  Native SQuAD2 questions  |  Previous adapter retained; a separate span guard passed validation", fontsize=11)
    initial = metrics["before"]["validation"]["generation"]
    values = [initial] + [row["validation"]["generation"] for row in trials]
    steps = [0] + [row["step"] for row in trials]
    axes[0].plot(steps, [100 * row["answerable_exact_match"] for row in values], "o-", color="#315f91", label="Answerable exact match")
    axes[0].plot(steps, [100 * row["correct_abstention_recall"] for row in values], "o-", color="#167864", label="Missing-answer abstention")
    axes[0].axhline(100 * (initial["answerable_exact_match"] - 0.05), color="#b25345", linestyle="--", linewidth=1.3, label="Minimum answer EM allowed")
    axes[0].set(title="Trained checkpoints failed retention", xlabel="Training update", ylabel="Validation score (%)", ylim=(0, 112), xticks=steps)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax, prefix, field, title in zip(axes[1:], ("answerable", "missing"),
                                      ("answerable_exact_match", "correct_abstention_recall"),
                                      ("Answerable test questions", "Missing-answer test questions")):
        for offset, phase, color, label in ((-0.18, "warmstart", "#315f91", "Previous adapter"),
                                          (0.18, "selected", "#167864", "Same adapter + span guard")):
            result = guard["test"][phase]["by_stratum"]
            values = [100 * result[f"{prefix}_{length}"][field] for length in ("short", "long")]
            ax.bar([offset, 1 + offset], values, width=0.32, color=color, label=label)
            for index, value in enumerate(values):
                ax.text(index + offset, value + 1.6, f"{value:.1f}", ha="center", fontsize=10)
        ax.set(xticks=[0, 1], xticklabels=["Short: ≤192 tokens", "Longer: 193–512"],
               ylabel="Correct (%)", ylim=(0, 112), yticks=[0, 20, 40, 60, 80, 100])
        ax.set_title(title, loc="left", fontsize=13, pad=15)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].text(0, -0.34, "128 validation questions; 640 training examples.\nAll four checkpoints failed the answer-retention gate.", transform=axes[0].transAxes, fontsize=9)
    axes[1].text(0, -0.34, "Exact answers retained: 66/128.\nAnswerable token F1: 58.63% → 56.35%.", transform=axes[1].transAxes, fontsize=9)
    axes[2].text(0, -0.34, "15/128 missing cases caught; most remain undetected.\n64 test questions per subgroup; complete paragraphs.", transform=axes[2].transAxes, fontsize=9)
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "document-v2.png", dpi=160)
    svg = folder / "document-v2.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
