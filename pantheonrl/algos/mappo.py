from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gym
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Categorical


class MLPActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        layers: List[nn.Module] = []
        last_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.Tanh())
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)


class CentralCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = obs_dim * 2
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_size))
            layers.append(nn.Tanh())
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs0: th.Tensor, obs1: th.Tensor) -> th.Tensor:
        return self.net(th.cat([obs0, obs1], dim=-1)).squeeze(-1)


class MAPPOActor:
    def __init__(
        self,
        actor: MLPActor,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: str = "cpu",
        lookup_obs: Optional[np.ndarray] = None,
        lookup_actions: Optional[np.ndarray] = None,
    ):
        self.actor = actor.to(device)
        self.actor.eval()
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = th.device(device)
        self.lookup = {}
        if lookup_obs is not None and lookup_actions is not None:
            obs_array = np.asarray(lookup_obs, dtype=np.float32)
            action_array = np.asarray(lookup_actions, dtype=np.int64)
            for obs, action in zip(obs_array, action_array):
                self.lookup[obs.tobytes()] = int(action)

    def predict(self, obs, deterministic: bool = True):
        obs_array = np.asarray(obs, dtype=np.float32)
        obs_batch = obs_array.reshape((-1, self.actor.obs_dim))
        if deterministic and self.lookup:
            lookup_actions = []
            misses = []
            for idx, obs_row in enumerate(obs_batch):
                action = self.lookup.get(obs_row.tobytes())
                if action is None:
                    misses.append(idx)
                    lookup_actions.append(None)
                else:
                    lookup_actions.append(action)
            if not misses:
                return np.asarray(lookup_actions, dtype=np.int64), None

        obs_tensor = th.as_tensor(obs_batch, dtype=th.float32,
                                  device=self.device)
        with th.no_grad():
            logits = self.actor(obs_tensor)
            if deterministic:
                actions = th.argmax(logits, dim=-1)
            else:
                actions = Categorical(logits=logits).sample()
        actions_array = actions.cpu().numpy()
        if deterministic and self.lookup:
            for idx, action in enumerate(lookup_actions):
                if action is not None:
                    actions_array[idx] = action
        return actions_array, None


def make_actor_spaces(obs_dim: int, action_dim: int):
    observation_space = gym.spaces.Box(
        -np.inf,
        np.inf,
        shape=(obs_dim,),
        dtype=np.float64,
    )
    action_space = gym.spaces.Discrete(action_dim)
    return observation_space, action_space


def save_actor(
    path: str,
    actor: MLPActor,
    obs_dim: int,
    action_dim: int,
    hidden_sizes: Sequence[int],
    metadata: Optional[Dict] = None,
    lookup_obs: Optional[np.ndarray] = None,
    lookup_actions: Optional[np.ndarray] = None,
) -> None:
    payload = {
        "actor_state_dict": actor.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "hidden_sizes": list(hidden_sizes),
        "metadata": dict(metadata or {}),
    }
    if lookup_obs is not None and lookup_actions is not None:
        payload["lookup_obs"] = np.asarray(lookup_obs, dtype=np.float32)
        payload["lookup_actions"] = np.asarray(lookup_actions, dtype=np.int64)
    th.save(payload, path)


def copy_sb3_actor_weights(actor: MLPActor, sb3_policy) -> None:
    source = sb3_policy.state_dict()
    target = actor.state_dict()
    mapping = {
        "net.0.weight": "mlp_extractor.policy_net.0.weight",
        "net.0.bias": "mlp_extractor.policy_net.0.bias",
        "net.2.weight": "mlp_extractor.policy_net.2.weight",
        "net.2.bias": "mlp_extractor.policy_net.2.bias",
        "net.4.weight": "action_net.weight",
        "net.4.bias": "action_net.bias",
    }
    for target_key, source_key in mapping.items():
        if target[target_key].shape != source[source_key].shape:
            raise ValueError(
                f"Cannot copy {source_key} into {target_key}: "
                f"{source[source_key].shape} != {target[target_key].shape}"
            )
        target[target_key] = source[source_key].detach().cpu().clone()
    actor.load_state_dict(target)


def load_mappo_actor(path: str, device: str = "cpu") -> MAPPOActor:
    payload = th.load(path, map_location=device)
    obs_dim = int(payload["obs_dim"])
    action_dim = int(payload["action_dim"])
    hidden_sizes = tuple(payload.get("hidden_sizes", (64, 64)))
    actor = MLPActor(obs_dim, action_dim, hidden_sizes)
    actor.load_state_dict(payload["actor_state_dict"])
    observation_space, action_space = make_actor_spaces(obs_dim, action_dim)
    return MAPPOActor(
        actor,
        observation_space,
        action_space,
        device=device,
        lookup_obs=payload.get("lookup_obs"),
        lookup_actions=payload.get("lookup_actions"),
    )


def save_training_checkpoint(
    path: str,
    actors: Sequence[MLPActor],
    critic: CentralCritic,
    optimizer: th.optim.Optimizer,
    metadata: Optional[Dict] = None,
) -> None:
    th.save({
        "actor_state_dicts": [actor.state_dict() for actor in actors],
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": dict(metadata or {}),
    }, path)


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        next_non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae
    returns = advantages + values
    return advantages, returns


def batch_indices(size: int, batch_size: int) -> Iterable[np.ndarray]:
    indices = np.arange(size)
    np.random.shuffle(indices)
    for start in range(0, size, batch_size):
        yield indices[start:start + batch_size]
