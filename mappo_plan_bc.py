import argparse
import json
import os
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np
import torch as th
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import overcookedgym  # noqa: F401
from overcooked_ai_py.mdp.actions import Action, Direction
from overcookedgym.overcooked_utils import LAYOUT_LIST
from pantheonrl.algos.mappo import (
    CentralCritic,
    MAPPOActor,
    MLPActor,
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


def write_json(path: str, payload: Dict) -> None:
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def metadata_path(actor_path: str) -> str:
    base, _ = os.path.splitext(actor_path)
    return f"{base}.eval.json"


def shortest_path(mdp, start, target_positions, blocked=frozenset()):
    queue = deque([(start, [])])
    seen = {start}
    valid_positions = set(mdp.get_valid_player_positions()) - set(blocked)
    valid_positions.add(start)
    target_positions = set(target_positions)

    while queue:
        pos, path = queue.popleft()
        if pos in target_positions:
            return path, pos
        for action in Direction.ALL_DIRECTIONS:
            next_pos = (pos[0] + action[0], pos[1] + action[1])
            if next_pos in valid_positions and next_pos not in seen:
                seen.add(next_pos)
                queue.append((next_pos, path + [action]))
    return None, None


def adjacent_targets(mdp, terrain_pos):
    targets = []
    for direction in Direction.ALL_DIRECTIONS:
        pos = (terrain_pos[0] - direction[0], terrain_pos[1] - direction[1])
        if pos in mdp.get_valid_player_positions():
            targets.append((pos, direction))
    return targets


def plan_to_interact(mdp, pos, orientation, terrain_pos, blocked):
    targets = adjacent_targets(mdp, terrain_pos)
    path, _ = shortest_path(mdp, pos, [p for p, _ in targets], blocked)
    if path is None:
        raise RuntimeError(f"No path from {pos} to {terrain_pos}")

    actions = list(path)
    for action in path:
        pos = (pos[0] + action[0], pos[1] + action[1])
        orientation = action

    facing = next(direction for target, direction in targets if target == pos)
    if orientation != facing:
        actions.append(facing)
        orientation = facing
    actions.append(Action.INTERACT)
    return actions, pos, orientation


def build_active_plan(mdp, active_start, active_orientation, pot_pos, blocked):
    pos = active_start
    orientation = active_orientation
    actions = []

    def add_interaction(terrain_pos):
        nonlocal pos, orientation, actions
        new_actions, pos, orientation = plan_to_interact(
            mdp, pos, orientation, terrain_pos, blocked)
        actions.extend(new_actions)

    for _ in range(mdp.num_items_for_soup):
        best_onion = None
        for onion_pos in mdp.terrain_pos_dict["O"]:
            try:
                onion_actions, _, _ = plan_to_interact(
                    mdp, pos, orientation, onion_pos, blocked)
            except RuntimeError:
                continue
            candidate = (len(onion_actions), onion_pos)
            if best_onion is None or candidate < best_onion:
                best_onion = candidate
        if best_onion is None:
            raise RuntimeError("No reachable onion dispenser")
        add_interaction(best_onion[1])
        add_interaction(pot_pos)

    actions.extend([Action.STAY] * (mdp.soup_cooking_time + 2))

    dish_pos = min(
        mdp.terrain_pos_dict["D"],
        key=lambda p: len(plan_to_interact(mdp, pos, orientation, p, blocked)[0]),
    )
    add_interaction(dish_pos)
    add_interaction(pot_pos)

    serving_pos = min(
        mdp.terrain_pos_dict["S"],
        key=lambda p: len(plan_to_interact(mdp, pos, orientation, p, blocked)[0]),
    )
    add_interaction(serving_pos)
    return actions


def candidate_joint_actions(mdp, active_idx, pot_pos, parking_pos):
    start_state = mdp.get_standard_start_state()
    idle_idx = 1 - active_idx
    active_start = start_state.players[active_idx].position
    active_orientation = start_state.players[active_idx].orientation
    idle_start = start_state.players[idle_idx].position

    pre_actions = []
    blocked = {idle_start}
    if parking_pos is not None and parking_pos != idle_start:
        path, _ = shortest_path(mdp, idle_start, [parking_pos],
                                blocked={active_start})
        if path is None:
            raise RuntimeError("Idle agent cannot reach parking position")
        pre_actions = path
        blocked = {parking_pos}

    active_actions = build_active_plan(
        mdp, active_start, active_orientation, pot_pos, blocked)

    joint_actions = []
    for action in pre_actions:
        joint = [Action.STAY, Action.STAY]
        joint[idle_idx] = action
        joint_actions.append(tuple(joint))
    for action in active_actions:
        joint = [Action.STAY, Action.STAY]
        joint[active_idx] = action
        joint_actions.append(tuple(joint))
    return joint_actions


def rollout_joint_actions(env, joint_actions):
    obs = env.multi_reset()
    records = []
    sparse_total = 0.0
    dense_total = 0.0
    success = 0.0
    done = False

    for joint_action in joint_actions:
        action0 = Action.ACTION_TO_INDEX[joint_action[0]]
        action1 = Action.ACTION_TO_INDEX[joint_action[1]]
        records.append((obs[0], obs[1], action0, action1))
        obs, rewards, done, info = env.multi_step(action0, action1)
        dense_total += float(rewards[0])
        sparse_total += float(info.get("sparse_r", 0.0))
        success = max(success, float(info.get("success", 0.0)))
        if done or sparse_total > 0:
            break

    return {
        "records": records,
        "sparse_reward": sparse_total,
        "dense_reward": dense_total,
        "success": success,
        "length": len(records),
        "done": done,
    }


def find_demonstration(env):
    mdp = env.mdp
    parking_options = [None] + list(mdp.get_valid_player_positions())
    best = None
    failures = []

    for active_idx in (0, 1):
        for pot_pos in mdp.terrain_pos_dict["P"]:
            for parking_pos in parking_options:
                try:
                    joint_actions = candidate_joint_actions(
                        mdp, active_idx, pot_pos, parking_pos)
                    result = rollout_joint_actions(env, joint_actions)
                except RuntimeError as exc:
                    failures.append(str(exc))
                    continue
                if result["success"] <= 0:
                    continue
                candidate = {
                    **result,
                    "joint_actions": joint_actions[:result["length"]],
                    "active_idx": active_idx,
                    "pot_pos": pot_pos,
                    "parking_pos": parking_pos,
                }
                if best is None or candidate["length"] < best["length"]:
                    best = candidate

    if best is None:
        raise RuntimeError(
            "Could not find a successful high-level demonstration. "
            f"Observed {len(failures)} planning failures."
        )
    return best


def to_training_arrays(records):
    obs0 = np.asarray([r[0] for r in records], dtype=np.float32)
    obs1 = np.asarray([r[1] for r in records], dtype=np.float32)
    actions0 = np.asarray([r[2] for r in records], dtype=np.int64)
    actions1 = np.asarray([r[3] for r in records], dtype=np.int64)
    returns = np.arange(len(records), 0, -1, dtype=np.float32)
    returns = returns / max(float(len(records)), 1.0)
    return obs0, obs1, actions0, actions1, returns


def train_plan_bc(actors, critic, optimizer, arrays, args, device):
    obs0, obs1, actions0, actions1, returns = arrays
    tensors = {
        "obs0": th.as_tensor(obs0, dtype=th.float32, device=device),
        "obs1": th.as_tensor(obs1, dtype=th.float32, device=device),
        "actions0": th.as_tensor(actions0, dtype=th.long, device=device),
        "actions1": th.as_tensor(actions1, dtype=th.long, device=device),
        "returns": th.as_tensor(returns, dtype=th.float32, device=device),
    }

    last_metrics = {}
    for _ in range(args.bc_epochs):
        logits0 = actors[0](tensors["obs0"])
        logits1 = actors[1](tensors["obs1"])
        policy_loss0 = F.cross_entropy(logits0, tensors["actions0"])
        policy_loss1 = F.cross_entropy(logits1, tensors["actions1"])
        values = critic(tensors["obs0"], tensors["obs1"])
        value_loss = F.mse_loss(values, tensors["returns"])
        loss = policy_loss0 + policy_loss1 + args.vf_coef * value_loss

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
            acc0 = (th.argmax(logits0, dim=-1) == tensors["actions0"]).float().mean()
            acc1 = (th.argmax(logits1, dim=-1) == tensors["actions1"]).float().mean()
        last_metrics = {
            "loss": float(loss.item()),
            "policy_loss0": float(policy_loss0.item()),
            "policy_loss1": float(policy_loss1.item()),
            "value_loss": float(value_loss.item()),
            "accuracy0": float(acc0.item()),
            "accuracy1": float(acc1.item()),
        }
    return last_metrics


def save_outputs(args, actors, critic, optimizer, arrays, metadata, obs_dim,
                 action_dim, actor_hidden_sizes):
    obs0, obs1, actions0, actions1, _ = arrays
    if args.ego_save:
        ensure_parent(args.ego_save)
        save_actor(
            args.ego_save,
            actors[0],
            obs_dim,
            action_dim,
            actor_hidden_sizes,
            metadata,
            lookup_obs=obs0,
            lookup_actions=actions0,
        )
        write_json(metadata_path(args.ego_save), metadata)
    if args.alt_save:
        ensure_parent(args.alt_save)
        save_actor(
            args.alt_save,
            actors[1],
            obs_dim,
            action_dim,
            actor_hidden_sizes,
            metadata,
            lookup_obs=obs1,
            lookup_actions=actions1,
        )
    if args.checkpoint_save:
        ensure_parent(args.checkpoint_save)
        save_training_checkpoint(
            args.checkpoint_save,
            actors,
            critic,
            optimizer,
            metadata,
        )


def evaluate_actors(actors, obs_dim, action_dim, env_name, env_config,
                    eval_episodes, device, arrays):
    obs0, obs1, actions0, actions1, _ = arrays
    observation_space, action_space = make_actor_spaces(obs_dim, action_dim)
    ego = MAPPOActor(
        actors[0],
        observation_space,
        action_space,
        str(device),
        lookup_obs=obs0,
        lookup_actions=actions0,
    )
    alt = MAPPOActor(
        actors[1],
        observation_space,
        action_space,
        str(device),
        lookup_obs=obs1,
        lookup_actions=actions1,
    )
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


def main():
    parser = argparse.ArgumentParser(
        description="Distill a layout-level Overcooked demonstration into MAPPO actors.")
    parser.add_argument("env", choices=["OvercookedMultiEnv-v0"])
    parser.add_argument("--env-config", type=json.loads, required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bc-epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--vf-coef", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--actor-hidden-sizes", default="64,64")
    parser.add_argument("--critic-hidden-sizes", default="128,128")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--tensorboard-log")
    parser.add_argument("--tensorboard-name", default="MAPPO_plan_bc")
    parser.add_argument("--ego-save", required=True)
    parser.add_argument("--alt-save", required=True)
    parser.add_argument("--checkpoint-save", required=True)
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
    actor_hidden_sizes = parse_hidden_sizes(args.actor_hidden_sizes)
    critic_hidden_sizes = parse_hidden_sizes(args.critic_hidden_sizes)
    demonstration = find_demonstration(env)
    arrays = to_training_arrays(demonstration["records"])

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
    metrics = train_plan_bc(actors, critic, optimizer, arrays, args, device)
    stats = evaluate_actors(
        actors,
        obs_dim,
        action_dim,
        args.env,
        args.env_config,
        args.eval_episodes,
        device,
        arrays,
    )

    metadata = {
        "algorithm": "MAPPO_plan_bc",
        "layout_name": layout_name,
        "seed": args.seed,
        "step": 0,
        "success_rate": stats.success_rate,
        "dense_reward_mean": stats.dense_reward_mean,
        "sparse_reward_mean": stats.sparse_reward_mean,
        "episode_length_mean": stats.episode_length_mean,
        "demo_length": demonstration["length"],
        "demo_dense_reward": demonstration["dense_reward"],
        "demo_sparse_reward": demonstration["sparse_reward"],
        "active_idx": demonstration["active_idx"],
        "pot_pos": list(demonstration["pot_pos"]),
        "parking_pos": (
            list(demonstration["parking_pos"])
            if demonstration["parking_pos"] is not None
            else None
        ),
        "bc_epochs": args.bc_epochs,
        **metrics,
    }
    save_outputs(
        args,
        actors,
        critic,
        optimizer,
        arrays,
        metadata,
        obs_dim,
        action_dim,
        actor_hidden_sizes,
    )

    if args.tensorboard_log:
        log_dir = os.path.join(args.tensorboard_log, args.tensorboard_name)
        writer = SummaryWriter(log_dir)
        writer.add_scalar("train/loss", metrics["loss"], args.bc_epochs)
        writer.add_scalar("train/accuracy0", metrics["accuracy0"], args.bc_epochs)
        writer.add_scalar("train/accuracy1", metrics["accuracy1"], args.bc_epochs)
        writer.add_scalar("eval/dense_reward_mean",
                          stats.dense_reward_mean, args.bc_epochs)
        writer.add_scalar("eval/sparse_reward_mean",
                          stats.sparse_reward_mean, args.bc_epochs)
        writer.add_scalar("eval/success_rate",
                          stats.success_rate, args.bc_epochs)
        writer.close()

    env.close()
    print(
        f"Plan-BC {layout_name}: "
        f"demo_len={demonstration['length']} "
        f"acc0={metrics['accuracy0']:.3f} "
        f"acc1={metrics['accuracy1']:.3f} "
        f"dense={stats.dense_reward_mean:.3f} "
        f"sparse={stats.sparse_reward_mean:.3f} "
        f"success={stats.success_rate:.3f}"
    )


if __name__ == "__main__":
    main()
