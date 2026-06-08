import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import gym
import numpy as np
import torch as th
import overcookedgym

from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

from pantheonrl.algos.adap.adap_learn import ADAP
from pantheonrl.algos.bc import BCShell, reconstruct_policy
from pantheonrl.algos.modular.learn import ModularAlgorithm
from pantheonrl.common.agents import StaticModelAgent, StaticPolicyAgent
from pantheonrl.common.wrappers import frame_wrap

TENSORBOARD_SCALAR_WHITELIST = {
    "eval/success_rate",
    "eval/dense_reward_mean",
    "rollout/ep_rew_mean",
}


@dataclass
class EvalStats:
    dense_reward_mean: float
    dense_reward_std: float
    sparse_reward_mean: float
    sparse_reward_std: float
    success_rate: float
    episode_length_mean: float


def load_policy(policy_type: str, location: str, config: Optional[dict] = None):
    config = dict(config or {})
    if policy_type == "PPO":
        return PPO.load(location)
    if policy_type == "A2C":
        return A2C.load(location)
    if policy_type == "DQN":
        return DQN.load(location)
    if policy_type == "SAC":
        return SAC.load(location)
    if policy_type == "ModularAlgorithm":
        return ModularAlgorithm.load(location).policy
    if policy_type in ("ADAP", "ADAP_MULT"):
        model = ADAP.load(location)
        if "latent_val" in config:
            model.policy.set_context(th.tensor(config["latent_val"]))
        return model.policy
    if policy_type == "BC":
        return BCShell(reconstruct_policy(location)).policy
    raise ValueError(f"Unsupported policy type: {policy_type}")


def policy_from_agent(agent):
    if hasattr(agent, "policy"):
        return agent.policy
    if hasattr(agent, "model"):
        return agent.model
    raise ValueError(f"Cannot extract policy from {agent}")


def static_agent(policy_or_model):
    if hasattr(policy_or_model, "predict"):
        return StaticModelAgent(policy_or_model)
    return StaticPolicyAgent(policy_or_model)


def make_eval_env(env_name: str, env_config: dict, framestack: int = 1):
    env = gym.make(env_name, **env_config)
    if framestack > 1:
        env = frame_wrap(env, framestack)
    env.set_ego_extractor(lambda obs: obs)
    return env


def evaluate_policy_pair(
    ego_policy,
    partner_policy,
    env_name: str,
    env_config: dict,
    episodes: int,
    framestack: int = 1,
) -> EvalStats:
    env = make_eval_env(env_name, env_config, framestack)
    ego = static_agent(ego_policy)
    partner = static_agent(partner_policy)
    env.add_partner_agent(partner)

    dense_rewards: List[float] = []
    sparse_rewards: List[float] = []
    successes: List[float] = []
    lengths: List[int] = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        dense_reward = 0.0
        sparse_reward = 0.0
        success = 0.0
        length = 0
        while not done:
            action = ego.get_action(obs, False)
            obs, reward, done, info = env.step(action)
            dense_reward += reward
            sparse_reward += info.get("sparse_r", 0.0)
            success = max(success, info.get("success", 0.0))
            length += 1

        if info.get("episode"):
            sparse_reward = info["episode"].get("ep_sparse_r", sparse_reward)
            success = info["episode"].get("success", success)
        dense_rewards.append(dense_reward)
        sparse_rewards.append(sparse_reward)
        successes.append(success)
        lengths.append(length)

    env.close()
    return EvalStats(
        dense_reward_mean=float(np.mean(dense_rewards)),
        dense_reward_std=float(np.std(dense_rewards)),
        sparse_reward_mean=float(np.mean(sparse_rewards)),
        sparse_reward_std=float(np.std(sparse_rewards)),
        success_rate=float(np.mean(successes)),
        episode_length_mean=float(np.mean(lengths)),
    )


def write_cross_play_csv(rows: Iterable[Dict], output_path: str) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def filter_tensorboard_scalars(logger, whitelist: Set[str] = None) -> None:
    if getattr(logger, "_pantheon_tb_filter_enabled", False):
        return

    whitelist = whitelist or TENSORBOARD_SCALAR_WHITELIST
    original_record = logger.record

    def record(key, value, exclude=None):
        if key not in whitelist:
            exclude = _add_tensorboard_exclude(exclude)
        return original_record(key, value, exclude=exclude)

    logger.record = record
    logger._pantheon_tb_filter_enabled = True


def _add_tensorboard_exclude(exclude):
    if exclude is None:
        return ("tensorboard",)
    if isinstance(exclude, str):
        return exclude if exclude == "tensorboard" else (exclude, "tensorboard")
    if "tensorboard" in exclude:
        return exclude
    return tuple(exclude) + ("tensorboard",)


class TensorboardScalarFilterCallback(BaseCallback):
    def _on_training_start(self) -> None:
        filter_tensorboard_scalars(self.logger)

    def _on_step(self) -> bool:
        return True


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        env_name: str,
        env_config: dict,
        partner_agents: List,
        episodes: int,
        eval_freq: int,
        framestack: int = 1,
        best_ego_save: Optional[str] = None,
        best_alt_save: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_name = env_name
        self.env_config = dict(env_config)
        self.partner_agents = partner_agents
        self.episodes = episodes
        self.eval_freq = eval_freq
        self.framestack = framestack
        self.best_ego_save = best_ego_save
        self.best_alt_save = best_alt_save
        self.best_success_rate = -1.0
        self.best_dense_reward = -float("inf")
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self.last_eval_step = self.num_timesteps

        rows = []
        for idx, partner_agent in enumerate(self.partner_agents):
            stats = evaluate_policy_pair(
                self.model,
                policy_from_agent(partner_agent),
                self.env_name,
                self.env_config,
                self.episodes,
                self.framestack,
            )
            rows.append(stats)
            prefix = f"eval/partner_{idx}"
            self.logger.record(f"{prefix}/dense_reward_mean",
                               stats.dense_reward_mean)
            self.logger.record(f"{prefix}/sparse_reward_mean",
                               stats.sparse_reward_mean)
            self.logger.record(f"{prefix}/success_rate", stats.success_rate)
            self.logger.record(f"{prefix}/episode_length_mean",
                               stats.episode_length_mean)

        if rows:
            dense_mean = float(np.mean([r.dense_reward_mean for r in rows]))
            sparse_mean = float(np.mean([r.sparse_reward_mean for r in rows]))
            success_rate = float(np.mean([r.success_rate for r in rows]))
            self.logger.record("eval/dense_reward_mean", dense_mean)
            self.logger.record("eval/sparse_reward_mean", sparse_mean)
            self.logger.record("eval/success_rate", success_rate)
            print(
                f"Eval step={self.num_timesteps} "
                f"dense={dense_mean:.3f} sparse={sparse_mean:.3f} "
                f"success={success_rate:.3f}"
            )
            self._save_best_checkpoint(success_rate, dense_mean, sparse_mean)
            self.logger.dump(step=self.num_timesteps)
        return True

    def _save_best_checkpoint(
        self,
        success_rate: float,
        dense_reward: float,
        sparse_reward: float,
    ) -> None:
        is_better = (
            success_rate > self.best_success_rate or
            (success_rate == self.best_success_rate and
             dense_reward > self.best_dense_reward)
        )
        if not is_better:
            return

        self.best_success_rate = success_rate
        self.best_dense_reward = dense_reward
        if self.best_ego_save:
            _save_model(self.model, self.best_ego_save)
            _write_metadata(self.best_ego_save, {
                "step": self.num_timesteps,
                "success_rate": success_rate,
                "dense_reward_mean": dense_reward,
                "sparse_reward_mean": sparse_reward,
            })
        if self.best_alt_save:
            _save_partner_models(self.partner_agents, self.best_alt_save)
        if self.best_ego_save or self.best_alt_save:
            print(
                f"Saved best checkpoint at step={self.num_timesteps} "
                f"success={success_rate:.3f}"
            )


def _save_partner_models(partner_agents: List, save_path: str) -> None:
    if len(partner_agents) == 1:
        model = getattr(partner_agents[0], "model", None)
        if model is not None:
            _save_model(model, save_path)
        return

    os.makedirs(save_path, exist_ok=True)
    for idx, partner in enumerate(partner_agents):
        model = getattr(partner, "model", None)
        if model is not None:
            _save_model(model, os.path.join(save_path, str(idx)))


def _save_model(model, save_path: str) -> None:
    dirname = os.path.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    model.save(save_path)


def _write_metadata(save_path: str, metadata: Dict) -> None:
    metadata_path = f"{save_path}.eval.json"
    dirname = os.path.dirname(metadata_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
