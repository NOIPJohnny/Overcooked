import argparse
import json
import os
from typing import Dict

import gym
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

import overcookedgym  # noqa: F401
from stable_baselines3 import PPO

from overcookedgym.overcooked_utils import LAYOUT_LIST
from pantheonrl.algos.mappo import (
    CentralCritic,
    MAPPOActor,
    MLPActor,
    copy_sb3_actor_weights,
    make_actor_spaces,
    save_actor,
    save_training_checkpoint,
)
from pantheonrl.common.evaluation import evaluate_policy_pair


def ensure_parent(path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def write_json(path: str, payload: Dict) -> None:
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def metadata_path(actor_path: str) -> str:
    base, _ = os.path.splitext(actor_path)
    return f"{base}.eval.json"


def device_from_arg(device: str) -> th.device:
    if device == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(device)


def main():
    parser = argparse.ArgumentParser(
        description="Warm-start MAPPO actors from saved PPO policies.")
    parser.add_argument("env", choices=["OvercookedMultiEnv-v0"])
    parser.add_argument("--env-config", type=json.loads, required=True)
    parser.add_argument("--ego-ppo", required=True)
    parser.add_argument("--alt-ppo", required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--ego-save", required=True)
    parser.add_argument("--alt-save", required=True)
    parser.add_argument("--checkpoint-save", required=True)
    parser.add_argument("--tensorboard-log")
    parser.add_argument("--tensorboard-name", default="MAPPO_warmstart")
    args = parser.parse_args()

    if "layout_name" not in args.env_config:
        raise ValueError("layout_name needed for OvercookedMultiEnv-v0")
    layout_name = args.env_config["layout_name"]
    if layout_name not in LAYOUT_LIST:
        raise ValueError(f"{layout_name} is not a valid layout")

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    device = device_from_arg(args.device)
    env = gym.make(args.env, **args.env_config)
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    env.close()

    ego_ppo = PPO.load(args.ego_ppo)
    alt_ppo = PPO.load(args.alt_ppo)
    actors = [
        MLPActor(obs_dim, action_dim).to(device),
        MLPActor(obs_dim, action_dim).to(device),
    ]
    copy_sb3_actor_weights(actors[0], ego_ppo.policy)
    copy_sb3_actor_weights(actors[1], alt_ppo.policy)
    critic = CentralCritic(obs_dim).to(device)
    optimizer = th.optim.Adam(
        list(actors[0].parameters())
        + list(actors[1].parameters())
        + list(critic.parameters()),
        lr=3e-4,
        eps=1e-5,
    )

    observation_space, action_space = make_actor_spaces(obs_dim, action_dim)
    ego = MAPPOActor(actors[0], observation_space, action_space, str(device))
    alt = MAPPOActor(actors[1], observation_space, action_space, str(device))
    stats = evaluate_policy_pair(
        ego,
        alt,
        args.env,
        args.env_config,
        args.eval_episodes,
    )
    metadata = {
        "step": 0,
        "success_rate": stats.success_rate,
        "dense_reward_mean": stats.dense_reward_mean,
        "sparse_reward_mean": stats.sparse_reward_mean,
        "layout_name": layout_name,
        "seed": args.seed,
        "warmstart_from": {
            "ego": args.ego_ppo,
            "alt": args.alt_ppo,
        },
    }

    for path, actor in ((args.ego_save, actors[0]), (args.alt_save, actors[1])):
        ensure_parent(path)
        save_actor(path, actor, obs_dim, action_dim, (64, 64), metadata)
    ensure_parent(args.checkpoint_save)
    save_training_checkpoint(
        args.checkpoint_save,
        actors,
        critic,
        optimizer,
        metadata,
    )
    write_json(metadata_path(args.ego_save), metadata)

    if args.tensorboard_log:
        log_dir = os.path.join(args.tensorboard_log, args.tensorboard_name)
        writer = SummaryWriter(log_dir)
        writer.add_scalar("eval/dense_reward_mean",
                          stats.dense_reward_mean, 0)
        writer.add_scalar("eval/sparse_reward_mean",
                          stats.sparse_reward_mean, 0)
        writer.add_scalar("eval/success_rate", stats.success_rate, 0)
        writer.close()

    print(
        f"Warm-started {layout_name}: "
        f"dense={stats.dense_reward_mean:.3f} "
        f"sparse={stats.sparse_reward_mean:.3f} "
        f"success={stats.success_rate:.3f}"
    )


if __name__ == "__main__":
    main()
