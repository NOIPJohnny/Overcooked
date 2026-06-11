import argparse
import json
import os
from collections import deque
from typing import Dict, Optional, Sequence

import gym
import numpy as np
import torch as th
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import overcookedgym  # noqa: F401
from overcookedgym.overcooked_utils import LAYOUT_LIST
from pantheonrl.algos.mappo import (
    CentralCritic,
    MAPPOActor,
    MLPActor,
    batch_indices,
    compute_gae,
    make_actor_spaces,
    save_actor,
    save_training_checkpoint,
)
from pantheonrl.common.evaluation import evaluate_policy_pair


def parse_hidden_sizes(value: str) -> Sequence[int]:
    return tuple(int(v) for v in value.split(",") if v)


def device_from_arg(device: str) -> th.device:
    if device == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(device)


def ensure_parent(path: Optional[str]) -> None:
    if not path:
        return
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def metadata_path(actor_path: str) -> str:
    base, _ = os.path.splitext(actor_path)
    return f"{base}.eval.json"


def write_json(path: str, payload: Dict) -> None:
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_actor_pair(
    ego_path: Optional[str],
    alt_path: Optional[str],
    checkpoint_path: Optional[str],
    actors,
    critic,
    optimizer,
    obs_dim: int,
    action_dim: int,
    actor_hidden_sizes: Sequence[int],
    metadata: Dict,
) -> None:
    if ego_path:
        ensure_parent(ego_path)
        save_actor(ego_path, actors[0], obs_dim, action_dim,
                   actor_hidden_sizes, metadata)
    if alt_path:
        ensure_parent(alt_path)
        save_actor(alt_path, actors[1], obs_dim, action_dim,
                   actor_hidden_sizes, metadata)
    if checkpoint_path:
        ensure_parent(checkpoint_path)
        save_training_checkpoint(
            checkpoint_path,
            actors,
            critic,
            optimizer,
            metadata,
        )


def evaluate_actors(
    actors,
    obs_dim: int,
    action_dim: int,
    env_name: str,
    env_config: Dict,
    eval_episodes: int,
    device: th.device,
):
    observation_space, action_space = make_actor_spaces(obs_dim, action_dim)
    ego = MAPPOActor(actors[0], observation_space, action_space, str(device))
    alt = MAPPOActor(actors[1], observation_space, action_space, str(device))
    stats = evaluate_policy_pair(
        ego,
        alt,
        env_name,
        env_config,
        eval_episodes,
    )
    for actor in actors:
        actor.train()
    return stats


def collect_rollout(
    env,
    actors,
    critic,
    obs,
    n_steps: int,
    device: th.device,
    episode_rewards,
    episode_lengths,
    current_episode,
):
    obs0, obs1 = obs
    rollout = {
        "obs0": [],
        "obs1": [],
        "actions0": [],
        "actions1": [],
        "log_probs0": [],
        "log_probs1": [],
        "rewards": [],
        "dones": [],
        "values": [],
    }

    for _ in range(n_steps):
        obs0_t = th.as_tensor(obs0, dtype=th.float32,
                              device=device).unsqueeze(0)
        obs1_t = th.as_tensor(obs1, dtype=th.float32,
                              device=device).unsqueeze(0)
        with th.no_grad():
            logits0 = actors[0](obs0_t)
            logits1 = actors[1](obs1_t)
            dist0 = Categorical(logits=logits0)
            dist1 = Categorical(logits=logits1)
            action0 = dist0.sample()
            action1 = dist1.sample()
            log_prob0 = dist0.log_prob(action0)
            log_prob1 = dist1.log_prob(action1)
            value = critic(obs0_t, obs1_t)

        next_obs, rewards, done, _ = env.multi_step(
            int(action0.item()),
            int(action1.item()),
        )
        reward = float(rewards[0])

        rollout["obs0"].append(np.asarray(obs0, dtype=np.float32))
        rollout["obs1"].append(np.asarray(obs1, dtype=np.float32))
        rollout["actions0"].append(int(action0.item()))
        rollout["actions1"].append(int(action1.item()))
        rollout["log_probs0"].append(float(log_prob0.item()))
        rollout["log_probs1"].append(float(log_prob1.item()))
        rollout["rewards"].append(reward)
        rollout["dones"].append(float(done))
        rollout["values"].append(float(value.item()))

        current_episode["reward"] += reward
        current_episode["length"] += 1

        if done:
            episode_rewards.append(current_episode["reward"])
            episode_lengths.append(current_episode["length"])
            current_episode["reward"] = 0.0
            current_episode["length"] = 0
            obs0, obs1 = env.multi_reset()
        else:
            obs0, obs1 = next_obs

    with th.no_grad():
        last_obs0 = th.as_tensor(obs0, dtype=th.float32,
                                 device=device).unsqueeze(0)
        last_obs1 = th.as_tensor(obs1, dtype=th.float32,
                                 device=device).unsqueeze(0)
        last_value = float(critic(last_obs0, last_obs1).item())

    return {
        key: np.asarray(value)
        for key, value in rollout.items()
    }, (obs0, obs1), last_value, current_episode


def update_mappo(
    rollout,
    last_value: float,
    actors,
    critic,
    optimizer,
    args,
    device: th.device,
):
    advantages, returns = compute_gae(
        rollout["rewards"].astype(np.float32),
        rollout["dones"].astype(np.float32),
        rollout["values"].astype(np.float32),
        last_value,
        args.gamma,
        args.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    tensors = {
        "obs0": th.as_tensor(rollout["obs0"], dtype=th.float32, device=device),
        "obs1": th.as_tensor(rollout["obs1"], dtype=th.float32, device=device),
        "actions0": th.as_tensor(rollout["actions0"], dtype=th.long, device=device),
        "actions1": th.as_tensor(rollout["actions1"], dtype=th.long, device=device),
        "log_probs0": th.as_tensor(rollout["log_probs0"], dtype=th.float32,
                                   device=device),
        "log_probs1": th.as_tensor(rollout["log_probs1"], dtype=th.float32,
                                   device=device),
        "advantages": th.as_tensor(advantages, dtype=th.float32, device=device),
        "returns": th.as_tensor(returns, dtype=th.float32, device=device),
    }

    losses = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }
    size = len(rollout["rewards"])

    for _ in range(args.n_epochs):
        for idx in batch_indices(size, args.batch_size):
            logits0 = actors[0](tensors["obs0"][idx])
            logits1 = actors[1](tensors["obs1"][idx])
            dist0 = Categorical(logits=logits0)
            dist1 = Categorical(logits=logits1)

            new_log_prob0 = dist0.log_prob(tensors["actions0"][idx])
            new_log_prob1 = dist1.log_prob(tensors["actions1"][idx])
            ratio0 = th.exp(new_log_prob0 - tensors["log_probs0"][idx])
            ratio1 = th.exp(new_log_prob1 - tensors["log_probs1"][idx])
            advantages_batch = tensors["advantages"][idx]

            loss0 = ppo_policy_loss(ratio0, advantages_batch, args.clip_range)
            loss1 = ppo_policy_loss(ratio1, advantages_batch, args.clip_range)
            policy_loss = 0.5 * (loss0 + loss1)

            values = critic(tensors["obs0"][idx], tensors["obs1"][idx])
            value_loss = F.mse_loss(values, tensors["returns"][idx])
            entropy = 0.5 * (dist0.entropy().mean() + dist1.entropy().mean())
            loss = (
                policy_loss
                + args.vf_coef * value_loss
                - args.ent_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(
                list(actors[0].parameters())
                + list(actors[1].parameters())
                + list(critic.parameters()),
                args.max_grad_norm,
            )
            optimizer.step()

            with th.no_grad():
                approx_kl0 = tensors["log_probs0"][idx] - new_log_prob0
                approx_kl1 = tensors["log_probs1"][idx] - new_log_prob1
                clip_frac0 = th.abs(ratio0 - 1.0) > args.clip_range
                clip_frac1 = th.abs(ratio1 - 1.0) > args.clip_range
                losses["policy_loss"].append(float(policy_loss.item()))
                losses["value_loss"].append(float(value_loss.item()))
                losses["entropy"].append(float(entropy.item()))
                losses["approx_kl"].append(
                    float(0.5 * (approx_kl0.mean() + approx_kl1.mean()).item())
                )
                losses["clip_fraction"].append(
                    float(0.5 * (
                        clip_frac0.float().mean()
                        + clip_frac1.float().mean()
                    ).item())
                )

    return {key: float(np.mean(value)) for key, value in losses.items()}


def ppo_policy_loss(ratio: th.Tensor, advantages: th.Tensor,
                    clip_range: float) -> th.Tensor:
    unclipped = ratio * advantages
    clipped = th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    return -th.min(unclipped, clipped).mean()


def main():
    parser = argparse.ArgumentParser(
        description="Train MAPPO with decentralized actors and a centralized critic.")
    parser.add_argument("env", choices=["OvercookedMultiEnv-v0"])
    parser.add_argument("--env-config", type=json.loads, required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--actor-hidden-sizes", default="64,64")
    parser.add_argument("--critic-hidden-sizes", default="128,128")
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--tensorboard-log")
    parser.add_argument("--tensorboard-name", default="MAPPO")
    parser.add_argument("--ego-save")
    parser.add_argument("--alt-save")
    parser.add_argument("--checkpoint-save")
    parser.add_argument("--best-ego-save")
    parser.add_argument("--best-alt-save")
    parser.add_argument("--best-checkpoint-save")
    args = parser.parse_args()

    if "layout_name" not in args.env_config:
        raise ValueError("layout_name needed for OvercookedMultiEnv-v0")
    if args.env_config["layout_name"] not in LAYOUT_LIST:
        raise ValueError(f"{args.env_config['layout_name']} is not a valid layout")

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    device = device_from_arg(args.device)
    env = gym.make(args.env, **args.env_config)
    if hasattr(env, "seed"):
        env.seed(args.seed)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    actor_hidden_sizes = parse_hidden_sizes(args.actor_hidden_sizes)
    critic_hidden_sizes = parse_hidden_sizes(args.critic_hidden_sizes)

    actors = [
        MLPActor(obs_dim, action_dim, actor_hidden_sizes).to(device),
        MLPActor(obs_dim, action_dim, actor_hidden_sizes).to(device),
    ]
    critic = CentralCritic(obs_dim, critic_hidden_sizes).to(device)
    optimizer = th.optim.Adam(
        list(actors[0].parameters())
        + list(actors[1].parameters())
        + list(critic.parameters()),
        lr=args.learning_rate,
        eps=1e-5,
    )

    writer = None
    if args.tensorboard_log:
        log_dir = os.path.join(args.tensorboard_log, args.tensorboard_name)
        writer = SummaryWriter(log_dir)

    obs = env.multi_reset()
    global_step = 0
    next_eval_step = args.eval_freq
    best_success_rate = -1.0
    best_dense_reward = -float("inf")
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    current_episode = {"reward": 0.0, "length": 0}

    while global_step < args.total_timesteps:
        steps_to_collect = min(args.n_steps, args.total_timesteps - global_step)
        rollout, obs, last_value, current_episode = collect_rollout(
            env,
            actors,
            critic,
            obs,
            steps_to_collect,
            device,
            episode_rewards,
            episode_lengths,
            current_episode,
        )
        global_step += steps_to_collect
        losses = update_mappo(
            rollout,
            last_value,
            actors,
            critic,
            optimizer,
            args,
            device,
        )

        if writer:
            for key, value in losses.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            if episode_rewards:
                writer.add_scalar(
                    "rollout/ep_rew_mean",
                    float(np.mean(episode_rewards)),
                    global_step,
                )
                writer.add_scalar(
                    "rollout/ep_len_mean",
                    float(np.mean(episode_lengths)),
                    global_step,
                )

        print(
            f"step={global_step} "
            f"policy_loss={losses['policy_loss']:.3f} "
            f"value_loss={losses['value_loss']:.3f}"
        )

        if args.eval_freq > 0 and global_step >= next_eval_step:
            stats = evaluate_actors(
                actors,
                obs_dim,
                action_dim,
                args.env,
                args.env_config,
                args.eval_episodes,
                device,
            )
            if writer:
                writer.add_scalar("eval/dense_reward_mean",
                                  stats.dense_reward_mean, global_step)
                writer.add_scalar("eval/sparse_reward_mean",
                                  stats.sparse_reward_mean, global_step)
                writer.add_scalar("eval/success_rate",
                                  stats.success_rate, global_step)
            print(
                f"Eval step={global_step} "
                f"dense={stats.dense_reward_mean:.3f} "
                f"sparse={stats.sparse_reward_mean:.3f} "
                f"success={stats.success_rate:.3f}"
            )
            is_better = (
                stats.success_rate > best_success_rate
                or (
                    stats.success_rate == best_success_rate
                    and stats.dense_reward_mean > best_dense_reward
                )
            )
            if is_better:
                best_success_rate = stats.success_rate
                best_dense_reward = stats.dense_reward_mean
                metadata = {
                    "step": global_step,
                    "success_rate": stats.success_rate,
                    "dense_reward_mean": stats.dense_reward_mean,
                    "sparse_reward_mean": stats.sparse_reward_mean,
                    "layout_name": args.env_config["layout_name"],
                    "seed": args.seed,
                }
                save_actor_pair(
                    args.best_ego_save,
                    args.best_alt_save,
                    args.best_checkpoint_save,
                    actors,
                    critic,
                    optimizer,
                    obs_dim,
                    action_dim,
                    actor_hidden_sizes,
                    metadata,
                )
                if args.best_ego_save:
                    write_json(metadata_path(args.best_ego_save), metadata)
                print(
                    f"Saved best checkpoint at step={global_step} "
                    f"success={stats.success_rate:.3f}"
                )
            while global_step >= next_eval_step:
                next_eval_step += args.eval_freq

    final_metadata = {
        "step": global_step,
        "layout_name": args.env_config["layout_name"],
        "seed": args.seed,
    }
    save_actor_pair(
        args.ego_save,
        args.alt_save,
        args.checkpoint_save,
        actors,
        critic,
        optimizer,
        obs_dim,
        action_dim,
        actor_hidden_sizes,
        final_metadata,
    )

    if writer:
        writer.close()
    env.close()


if __name__ == "__main__":
    main()
