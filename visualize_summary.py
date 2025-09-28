import os
import json
import argparse
from typing import Dict, List, Tuple

import matplotlib

# Use a non-interactive backend to allow saving figures in headless environments (e.g., Colab/servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_summary(input_path: str) -> Tuple[int, Dict[str, int], Dict[str, float]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "summary" not in data:
        raise ValueError("Input JSON missing 'summary' block (expected output from classify_responses.py)")
    summary = data["summary"]
    total = int(summary.get("total_inputs", 0))
    counts = dict(summary.get("category_counts", {}))
    percents = dict(summary.get("category_percents", {}))
    return total, counts, percents


def ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def sort_categories(counts: Dict[str, int]) -> List[str]:
    # Sort by count desc, then alphabetically for stability
    return sorted(counts.keys(), key=lambda k: (-counts.get(k, 0), k.lower()))


def save_bar_counts(categories: List[str], counts: Dict[str, int], out_path: str, title: str) -> None:
    values = [counts.get(cat, 0) for cat in categories]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(categories) + 1)))
    ax.barh(categories, values, color="#4C78A8")
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.invert_yaxis()

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01 if values else 0.1, i, str(v), va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_bar_percents(categories: List[str], percents: Dict[str, float], out_path: str, title: str) -> None:
    values = [percents.get(cat, 0.0) for cat in categories]
    fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(categories) + 1)))
    ax.barh(categories, values, color="#F58518")
    ax.set_xlabel("Percent of Inputs (%)")
    ax.set_title(title)
    ax.invert_yaxis()

    for i, v in enumerate(values):
        ax.text(v + (max(values) * 0.01 if values else 1), i, f"{v:.2f}%", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_pie(categories: List[str], counts: Dict[str, int], out_path: str, title: str) -> None:
    values = [counts.get(cat, 0) for cat in categories]
    if sum(values) == 0:
        # Avoid empty pie warnings; create an empty figure with message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(values, labels=categories, autopct="%1.1f%%", startangle=140, textprops={"fontsize": 9})
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize summary from classify_responses.py output JSON.")
    parser.add_argument("--in", dest="input_path", required=True, help="Path to classified_responses.json")
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        default="figures",
        help="Directory to write PNGs (will be created if needed)",
    )
    parser.add_argument(
        "--prefix",
        dest="prefix",
        default="summary",
        help="Filename prefix for saved figures",
    )
    args = parser.parse_args()

    total, counts, percents = load_summary(args.input_path)
    out_dir = ensure_out_dir(args.out_dir)
    cats_sorted = sort_categories(counts)

    counts_path = os.path.join(out_dir, f"{args.prefix}_category_counts.png")
    save_bar_counts(cats_sorted, counts, counts_path, f"Category Counts (n={total})")

    percents_path = os.path.join(out_dir, f"{args.prefix}_category_percents.png")
    save_bar_percents(cats_sorted, percents, percents_path, "Category Percents")

    pie_path = os.path.join(out_dir, f"{args.prefix}_category_pie.png")
    save_pie(cats_sorted, counts, pie_path, f"Category Distribution (n={total})")

    print("Saved:")
    print("-", counts_path)
    print("-", percents_path)
    print("-", pie_path)


if __name__ == "__main__":
    main()


