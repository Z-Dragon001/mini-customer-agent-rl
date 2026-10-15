"""Render recorded measurements with matplotlib; no model/API calls."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/retail-3b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def main():
    runs = json.loads((OUT / "runs.json").read_text())
    if runs.get("sft"):
        path = ROOT / runs["sft"] / "sft_metrics.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        if rows:
            x = list(range(1, len(rows) + 1))
            losses = [r["loss"] for r in rows]
            smooth = [sum(losses[max(0, i-9):i+1]) / len(losses[max(0, i-9):i+1]) for i in range(len(rows))]
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
            axes[0].plot(x, losses, color="#93B4D3", alpha=.65, label="Batch loss")
            axes[0].plot(x, smooth, color="#174C80", label="10-batch moving mean")
            axes[0].axvline(58.5, color="gray", linestyle="--", alpha=.5)
            axes[0].set(ylabel="Training loss", title="Qwen2.5-3B: assistant-response SFT")
            axes[0].legend(frameon=False)
            axes[1].plot(x, [r["learning_rate"] for r in rows], color="#257768")
            axes[1].set(xlabel="Optimizer batch", ylabel="Learning rate")
            fig.savefig(OUT / "sft_training.png", dpi=180)
            plt.close(fig)
    if runs.get("grpo"):
        path = ROOT / runs["grpo"] / "updates.jsonl"
        groups = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        if groups:
            fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
            x = [r["iteration"] + 1 for r in groups]
            ax.bar(x, [sum(r["rewards"]) for r in groups],
                   color=["#247EAA" if r["updated"] else "#AEBBC7" for r in groups])
            ax.scatter(x, [sum(r["rewards"]) for r in groups], color="#344A5E", s=12, zorder=3)
            ax.set(xlabel="Rollout group (different training tasks)", ylabel="Successful rollouts / 4",
                   yticks=range(5), ylim=(0, 4.6), title="GRPO sampling: blue = policy update, gray = no reward contrast")
            fig.savefig(OUT / "grpo_sampling.png", dpi=180)
            plt.close(fig)
    rows = list(csv.DictReader((OUT / "results.csv").open()))
    rows = [r for r in rows if r["split"] == "test" and r["evaluated"] == r["expected"]]
    if rows:
        names = {"base": "Base", "sft_eval": "SFT", "grpo_eval": "SFT + GRPO"}
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        bars = ax.bar([names[r["stage"]] for r in rows], [100*float(r["success_rate"]) for r in rows],
                      color=["#8297AA", "#247EAA", "#2B8C72"][:len(rows)], width=.55)
        ax.bar_label(bars, labels=[f"{r['successes']}/{r['evaluated']} ({100*float(r['success_rate']):.1f}%)" for r in rows], padding=5)
        ax.set(ylim=(0, 100), ylabel="Task success (%)", title="Retail held-out test: 40 tasks, one seed")
        fig.savefig(OUT / "test_success.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
