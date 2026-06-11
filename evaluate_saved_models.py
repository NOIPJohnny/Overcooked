import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional, Tuple

from pantheonrl.common.evaluation import evaluate_policy_pair, load_policy


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


def algorithm_spec(algorithm: str) -> Tuple[str, str, str]:
    if algorithm == "DQN_improved":
        return "DQN", "PPO", ".zip"
    if algorithm == "MAPPO":
        return "MAPPO", "MAPPO", ".pt"
    return algorithm, algorithm, ".zip"


def best_or_final(models_dir: Path, name: str, suffix: str) -> Optional[Path]:
    best = models_dir / f"{name}-best{suffix}"
    final = models_dir / f"{name}-final{suffix}"
    if best.exists():
        return best
    if final.exists():
        return final
    return None


def load_location(path: Path, suffix: str) -> str:
    if suffix == ".zip":
        return str(path.with_suffix(""))
    return str(path)


def model_pair(results_dir: Path, algorithm: str, layout: str):
    ego_type, partner_type, suffix = algorithm_spec(algorithm)
    models_dir = results_dir / algorithm / layout / "models"
    ego_path = best_or_final(models_dir, "ego", suffix)
    alt_path = best_or_final(models_dir, "alt", suffix)
    if ego_path is None or alt_path is None:
        return None
    return {
        "ego_type": ego_type,
        "partner_type": partner_type,
        "ego_path": ego_path,
        "partner_path": alt_path,
        "suffix": suffix,
    }


def ensure_parent(path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def write_csv(path: str, rows) -> None:
    ensure_parent(path)
    fieldnames = [
        "algorithm",
        "layout",
        "dense_reward_mean",
        "dense_reward_std",
        "sparse_reward_mean",
        "sparse_reward_std",
        "success_rate",
        "episode_length_mean",
        "episodes",
        "ego_path",
        "partner_path",
    ]
    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved Overcooked policy pairs with one metric schema.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--env", default="OvercookedMultiEnv-v0")
    parser.add_argument("--layouts", nargs="+", default=DEFAULT_LAYOUTS)
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--framestack", type=int, default=1)
    parser.add_argument("--output-csv", default="results/comparison/final_eval.csv")
    parser.add_argument("--output-json", default="results/comparison/final_eval.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for algorithm in args.algorithms:
        for layout in args.layouts:
            pair = model_pair(results_dir, algorithm, layout)
            if pair is None:
                print(f"Skipping {algorithm}/{layout}: missing model pair")
                continue

            ego = load_policy(
                pair["ego_type"],
                load_location(pair["ego_path"], pair["suffix"]),
            )
            partner = load_policy(
                pair["partner_type"],
                load_location(pair["partner_path"], pair["suffix"]),
            )
            stats = evaluate_policy_pair(
                ego,
                partner,
                args.env,
                {"layout_name": layout},
                args.episodes,
                args.framestack,
            )
            row = {
                "algorithm": algorithm,
                "layout": layout,
                "dense_reward_mean": stats.dense_reward_mean,
                "dense_reward_std": stats.dense_reward_std,
                "sparse_reward_mean": stats.sparse_reward_mean,
                "sparse_reward_std": stats.sparse_reward_std,
                "success_rate": stats.success_rate,
                "episode_length_mean": stats.episode_length_mean,
                "episodes": args.episodes,
                "ego_path": str(pair["ego_path"]),
                "partner_path": str(pair["partner_path"]),
            }
            rows.append(row)
            print(row)

    write_csv(args.output_csv, rows)
    ensure_parent(args.output_json)
    with open(args.output_json, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
