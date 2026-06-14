import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_LAYOUTS = [
    "corridor",
    "five_by_five",
    "random0",
    "random1",
    "random2",
    "random3",
    "scenario1_s",
    "scenario2",
    "scenario2_s",
    "scenario3",
    "scenario4",
    "schelling",
    "schelling_s",
    "small_corridor",
    "unident",
    "unident_s",
]

DEFAULT_ALGORITHMS = ["PPO", "DQN", "A2C", "DQN_improved", "MAPPO"]
DELIVERY_REWARD = 20.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def event_files(results_dir: Path, algorithm: str, layout: str):
    logs_dir = results_dir / algorithm / layout / "logs"
    if not logs_dir.exists():
        return []
    return [
        path
        for path in logs_dir.rglob("events.out.tfevents.*")
        if "_alt_" not in str(path)
    ]


def scalar_series(
    results_dir: Path,
    algorithm: str,
    layout: str,
    tag: str,
) -> List[Tuple[int, float]]:
    points: Dict[int, float] = {}
    for path in event_files(results_dir, algorithm, layout):
        try:
            accumulator = EventAccumulator(str(path))
            accumulator.Reload()
            if tag not in accumulator.Tags().get("scalars", []):
                continue
            for event in accumulator.Scalars(tag):
                points[event.step] = event.value
        except Exception as exc:
            print(f"Skipping event file {path}: {exc}")
    return sorted(points.items())


def eval_json(results_dir: Path, algorithm: str, layout: str) -> Optional[Dict]:
    path = results_dir / algorithm / layout / "models" / "ego-best.eval.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def plot_learning_curves(
    results_dir: Path,
    output_dir: Path,
    algorithms: List[str],
    layouts: List[str],
) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=False, sharey=True)
    axes = axes.flatten()
    for ax, layout in zip(axes, layouts):
        for algorithm in algorithms:
            points = scalar_series(
                results_dir,
                algorithm,
                layout,
                "eval/success_rate",
            )
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, label=algorithm, linewidth=1.8)
        ax.set_title(layout, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
    for ax in axes[len(layouts):]:
        ax.axis("off")
    axes[0].set_ylabel("eval/success_rate")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "learning_curves_success_rate.png", dpi=180)
    plt.close(fig)


def plot_final_success(
    results_dir: Path,
    output_dir: Path,
    algorithms: List[str],
    layouts: List[str],
) -> None:
    values = np.full((len(algorithms), len(layouts)), np.nan)
    for algo_idx, algorithm in enumerate(algorithms):
        for layout_idx, layout in enumerate(layouts):
            data = eval_json(results_dir, algorithm, layout)
            if data is not None:
                values[algo_idx, layout_idx] = data.get("success_rate", np.nan)

    fig, ax = plt.subplots(figsize=(18, 4.8))
    image = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels(algorithms)
    ax.set_xticks(range(len(layouts)))
    ax.set_xticklabels(layouts, rotation=45, ha="right")
    ax.set_title("Best eval success rate")
    for y in range(len(algorithms)):
        for x in range(len(layouts)):
            if np.isfinite(values[y, x]):
                ax.text(x, y, f"{values[y, x]:.1f}", ha="center",
                        va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="success_rate")
    fig.tight_layout()
    fig.savefig(output_dir / "final_success_rate.png", dpi=180)
    plt.close(fig)


def plot_final_rewards(
    results_dir: Path,
    output_dir: Path,
    algorithms: List[str],
    layouts: List[str],
) -> None:
    dense_means = []
    sparse_means = []
    for algorithm in algorithms:
        dense = []
        sparse = []
        for layout in layouts:
            data = eval_json(results_dir, algorithm, layout)
            if data is None:
                continue
            dense.append(data.get("dense_reward_mean", np.nan))
            sparse.append(data.get("sparse_reward_mean", np.nan))
        dense_means.append(float(np.nanmean(dense)) if dense else np.nan)
        sparse_means.append(float(np.nanmean(sparse)) if sparse else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(algorithms))
    axes[0].bar(x, dense_means)
    axes[0].set_title("Mean best dense reward")
    axes[1].bar(x, sparse_means)
    axes[1].set_title("Mean best sparse reward")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "final_dense_sparse_reward.png", dpi=180)
    plt.close(fig)


def plot_sparse_delivery_by_layout(
    results_dir: Path,
    output_dir: Path,
    algorithms: List[str],
    layouts: List[str],
) -> None:
    sparse_values = np.full((len(algorithms), len(layouts)), np.nan)
    for algo_idx, algorithm in enumerate(algorithms):
        for layout_idx, layout in enumerate(layouts):
            data = eval_json(results_dir, algorithm, layout)
            if data is not None:
                sparse_values[algo_idx, layout_idx] = data.get(
                    "sparse_reward_mean", np.nan)

    delivery_values = sparse_values / DELIVERY_REWARD
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(max(10, len(layouts) * 1.4), 7.0),
        sharex=True,
    )
    panels = [
        (axes[0], sparse_values, "Best eval sparse reward", "sparse_reward"),
        (axes[1], delivery_values, "Estimated deliveries", "deliveries"),
    ]

    for ax, values, title, label in panels:
        finite_values = values[np.isfinite(values)]
        vmax = float(np.max(finite_values)) if finite_values.size else 1.0
        vmax = max(vmax, 1.0)
        image = ax.imshow(values, aspect="auto", vmin=0, vmax=vmax,
                          cmap="viridis")
        ax.set_title(title)
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels(algorithms)
        for y in range(len(algorithms)):
            for x in range(len(layouts)):
                if np.isfinite(values[y, x]):
                    text = f"{values[y, x]:.0f}" if label == "sparse_reward" else f"{values[y, x]:.1f}"
                    ax.text(x, y, text, ha="center", va="center",
                            color="white", fontsize=8)
        fig.colorbar(image, ax=ax, label=label)

    axes[-1].set_xticks(range(len(layouts)))
    axes[-1].set_xticklabels(layouts, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "final_sparse_delivery_by_layout.png", dpi=180)
    plt.close(fig)


def action_counts(path: Path) -> Optional[Counter]:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return Counter(step.get("ego_action_name") for step in data.get("steps", []))


def plot_action_distribution(
    results_dir: Path,
    output_dir: Path,
    algorithms: List[str],
    layouts: List[str],
) -> None:
    gifs_dir = results_dir / "gifs"
    for layout in layouts:
        rows = []
        action_names = set()
        for algorithm in algorithms:
            counts = action_counts(gifs_dir / f"{algorithm}_{layout}.json")
            if not counts:
                continue
            rows.append((algorithm, counts))
            action_names.update(counts.keys())
        if not rows:
            continue

        ordered_actions = sorted(action_names)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bottom = np.zeros(len(rows))
        x = np.arange(len(rows))
        totals = np.array([sum(counts.values()) for _, counts in rows])
        for action_name in ordered_actions:
            values = np.array([
                counts.get(action_name, 0) / max(total, 1)
                for (_, counts), total in zip(rows, totals)
            ])
            ax.bar(x, values, bottom=bottom, label=action_name)
            bottom += values
        ax.set_title(f"Final policy ego action distribution: {layout}")
        ax.set_xticks(x)
        ax.set_xticklabels([algorithm for algorithm, _ in rows],
                           rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"action_distribution_{layout}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Overcooked learning curves and final policy comparisons.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--layouts", nargs="+", default=DEFAULT_LAYOUTS)
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    plot_learning_curves(results_dir, output_dir, args.algorithms, args.layouts)
    plot_final_success(results_dir, output_dir, args.algorithms, args.layouts)
    plot_final_rewards(results_dir, output_dir, args.algorithms, args.layouts)
    plot_sparse_delivery_by_layout(
        results_dir, output_dir, args.algorithms, args.layouts)
    plot_action_distribution(results_dir, output_dir, args.algorithms, args.layouts)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
