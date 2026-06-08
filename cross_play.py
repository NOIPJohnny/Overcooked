import argparse
import json

from pantheonrl.common.evaluation import (
    evaluate_policy_pair,
    load_policy,
    write_cross_play_csv,
)


def label(config, index):
    return config.get("name") or f'{config["type"]}:{config["location"]}:{index}'


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a cross-play matrix for saved policies.")
    parser.add_argument("env", help="Gym environment id")
    parser.add_argument("--env-config", type=json.loads, default={})
    parser.add_argument("--ego-config", type=json.loads, action="append",
                        required=True)
    parser.add_argument("--partner-config", type=json.loads, action="append",
                        required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--framestack", type=int, default=1)
    parser.add_argument("--output", default="cross_play.csv")
    parser.add_argument("--tensorboard-log")
    args = parser.parse_args()

    egos = [
        (label(config, idx),
         load_policy(config["type"], config["location"], config))
        for idx, config in enumerate(args.ego_config)
    ]
    partners = [
        (label(config, idx),
         load_policy(config["type"], config["location"], config))
        for idx, config in enumerate(args.partner_config)
    ]

    writer = None
    if args.tensorboard_log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tensorboard_log)

    rows = []
    for ego_name, ego_policy in egos:
        for partner_name, partner_policy in partners:
            stats = evaluate_policy_pair(
                ego_policy,
                partner_policy,
                args.env,
                args.env_config,
                args.episodes,
                args.framestack,
            )
            row = {
                "ego": ego_name,
                "partner": partner_name,
                "dense_reward_mean": stats.dense_reward_mean,
                "dense_reward_std": stats.dense_reward_std,
                "sparse_reward_mean": stats.sparse_reward_mean,
                "sparse_reward_std": stats.sparse_reward_std,
                "success_rate": stats.success_rate,
                "episode_length_mean": stats.episode_length_mean,
                "episodes": args.episodes,
            }
            rows.append(row)
            print(row)

            if writer is not None:
                tag = f"cross_play/{ego_name}_vs_{partner_name}"
                writer.add_scalar(f"{tag}/dense_reward_mean",
                                  stats.dense_reward_mean, 0)
                writer.add_scalar(f"{tag}/sparse_reward_mean",
                                  stats.sparse_reward_mean, 0)
                writer.add_scalar(f"{tag}/success_rate",
                                  stats.success_rate, 0)

    write_cross_play_csv(rows, args.output)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
