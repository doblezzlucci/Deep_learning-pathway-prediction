import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------
# Load CSVs
# --------------------------------------------------------
def visualize_results(paths):
  

  results = {}
  for name, paf in paths.items():
    df = pd.read_csv(paf)
    results[name] = {
        "proportions": df["proportion"].values,
        "mean": df["mean"].values,
        "std": df["std"].values,
    }

# --------------------------------------------------------
# Publication-ready Matplotlib style
# --------------------------------------------------------
  plt.style.use("seaborn-v0_8-whitegrid")
  plt.figure(figsize=(9, 6), dpi=300)

# Define colors for consistency
  colors = {
    "Baseline": "#1f77b4",
    "Fine-tuned (Frozen)": "#d62728",
    "Fine-tuned (Unfrozen)": "#2ca02c",
  }

# --------------------------------------------------------
# Plot curves
# --------------------------------------------------------
  for name, data in results.items():
    proportions = data["proportions"]
    mean = data["mean"]
    std = data["std"]

    plt.plot(
        proportions, mean,
        marker="o",
        linewidth=2.2,
        markersize=7,
        label=name,
        color=colors[name]
    )

    plt.fill_between(
        proportions,
        mean - std,
        mean + std,
        alpha=0.20,
        color=colors[name]
    )

# --------------------------------------------------------
# Axis labels & titles
# --------------------------------------------------------
  plt.xlabel("Training set proportion", fontsize=14)
  plt.ylabel("Test accuracy", fontsize=14)
  plt.title("Baseline vs Fine-tuned SSL Encoder — Accuracy Comparison", fontsize=16)

  plt.xticks(
    results[list(results.keys())[0]]["proportions"],
    [f"{int(p*100)}%" for p in results[list(results.keys())[0]]["proportions"]],
    fontsize=12, rotation=45
  )
  plt.yticks(fontsize=12)

  plt.ylim(0, 1.05)
  plt.legend(fontsize=12)

  plt.tight_layout()

# --------------------------------------------------------
# Save publication-ready Figure
# --------------------------------------------------------
# plt.savefig("accuracy_comparison.png", dpi=300)
# plt.savefig("accuracy_comparison.svg")  # vector format

  plt.show()
