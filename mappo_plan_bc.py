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


def advance_motion(pos, orientation, actions):
    for action in actions:
        pos = (pos[0] + action[0], pos[1] + action[1])
        orientation = action
    return pos, orientation


def plan_motion_to(mdp, pos, orientation, target_pos, blocked):
    path, _ = shortest_path(mdp, pos, [target_pos], blocked)
    if path is None:
        raise RuntimeError(f"No path from {pos} to parking {target_pos}")
    pos, orientation = advance_motion(pos, orientation, path)
    return path, pos, orientation


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


def build_fill_remaining_plan(mdp, start_pos, start_orientation, pot_pos,
                              blocked, item_count):
    pos = start_pos
    orientation = start_orientation
    actions = []

    def add_interaction(terrain_pos):
        nonlocal pos, orientation, actions
        new_actions, pos, orientation = plan_to_interact(
            mdp, pos, orientation, terrain_pos, blocked)
        actions.extend(new_actions)

    for _ in range(item_count):
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
    return actions, pos, orientation


def build_fill_plan(mdp, start_pos, start_orientation, pot_pos, blocked):
    return build_fill_remaining_plan(
        mdp,
        start_pos,
        start_orientation,
        pot_pos,
        blocked,
        mdp.num_items_for_soup,
    )


def build_serve_plan(mdp, start_pos, start_orientation, pot_pos, blocked):
    pos = start_pos
    orientation = start_orientation
    actions = []

    def add_interaction(terrain_pos):
        nonlocal pos, orientation, actions
        new_actions, pos, orientation = plan_to_interact(
            mdp, pos, orientation, terrain_pos, blocked)
        actions.extend(new_actions)

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
    return actions, pos, orientation


def build_dish_parking_plan(mdp, start_pos, start_orientation, parking_pos,
                            blocked):
    pos = start_pos
    orientation = start_orientation
    actions = []

    dish_pos = min(
        mdp.terrain_pos_dict["D"],
        key=lambda p: len(plan_to_interact(mdp, pos, orientation, p, blocked)[0]),
    )
    dish_actions, pos, orientation = plan_to_interact(
        mdp, pos, orientation, dish_pos, blocked)
    actions.extend(dish_actions)

    if parking_pos is not None and parking_pos != pos:
        parking_actions, pos, orientation = plan_motion_to(
            mdp, pos, orientation, parking_pos, blocked)
        actions.extend(parking_actions)

    return actions, pos, orientation


def build_soup_delivery_plan(mdp, start_pos, start_orientation, pot_pos,
                             blocked):
    pos = start_pos
    orientation = start_orientation
    actions = []

    pot_actions, pos, orientation = plan_to_interact(
        mdp, pos, orientation, pot_pos, blocked)
    actions.extend(pot_actions)

    serving_pos = min(
        mdp.terrain_pos_dict["S"],
        key=lambda p: len(plan_to_interact(mdp, pos, orientation, p, blocked)[0]),
    )
    serving_actions, pos, orientation = plan_to_interact(
        mdp, pos, orientation, serving_pos, blocked)
    actions.extend(serving_actions)
    return actions, pos, orientation


def build_deliver_held_soup_plan(mdp, start_pos, start_orientation, blocked):
    pos = start_pos
    orientation = start_orientation
    serving_pos = min(
        mdp.terrain_pos_dict["S"],
        key=lambda p: len(plan_to_interact(mdp, pos, orientation, p, blocked)[0]),
    )
    return plan_to_interact(mdp, pos, orientation, serving_pos, blocked)


def cooperative_joint_actions(mdp, cook_idx, pot_pos, server_parking,
                              cook_parking):
    start_state = mdp.get_standard_start_state()
    serve_idx = 1 - cook_idx
    cook_pos = start_state.players[cook_idx].position
    cook_orientation = start_state.players[cook_idx].orientation
    serve_pos = start_state.players[serve_idx].position
    serve_orientation = start_state.players[serve_idx].orientation

    joint_actions = []
    if server_parking is not None and server_parking != serve_pos:
        serve_actions, serve_pos, serve_orientation = plan_motion_to(
            mdp, serve_pos, serve_orientation, server_parking,
            blocked={cook_pos})
        for action in serve_actions:
            joint = [Action.STAY, Action.STAY]
            joint[serve_idx] = action
            joint_actions.append(tuple(joint))

    fill_actions, cook_pos, cook_orientation = build_fill_plan(
        mdp, cook_pos, cook_orientation, pot_pos, blocked={serve_pos})
    for action in fill_actions:
        joint = [Action.STAY, Action.STAY]
        joint[cook_idx] = action
        joint_actions.append(tuple(joint))

    if cook_parking is not None and cook_parking != cook_pos:
        cook_actions, cook_pos, cook_orientation = plan_motion_to(
            mdp, cook_pos, cook_orientation, cook_parking,
            blocked={serve_pos})
        for action in cook_actions:
            joint = [Action.STAY, Action.STAY]
            joint[cook_idx] = action
            joint_actions.append(tuple(joint))

    for _ in range(mdp.soup_cooking_time + 2):
        joint_actions.append((Action.STAY, Action.STAY))

    serve_actions, _, _ = build_serve_plan(
        mdp, serve_pos, serve_orientation, pot_pos, blocked={cook_pos})
    for action in serve_actions:
        joint = [Action.STAY, Action.STAY]
        joint[serve_idx] = action
        joint_actions.append(tuple(joint))

    return joint_actions


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


def rollout_joint_actions(env, joint_actions, horizon=None):
    obs = env.multi_reset()
    records = []
    sparse_total = 0.0
    dense_total = 0.0
    success = 0.0
    success_steps = []
    delivery_count = 0
    done = False

    for joint_action in joint_actions:
        if horizon is not None and len(records) >= horizon:
            break
        action0 = Action.ACTION_TO_INDEX[joint_action[0]]
        action1 = Action.ACTION_TO_INDEX[joint_action[1]]
        records.append((obs[0], obs[1], action0, action1))
        obs, rewards, done, info = env.multi_step(action0, action1)
        dense_total += float(rewards[0])
        sparse_r = float(info.get("sparse_r", 0.0))
        sparse_total += sparse_r
        success = max(success, float(info.get("success", 0.0)))
        if sparse_r > 0:
            success_steps.append(len(records) - 1)
            delivery_count += max(1, int(round(sparse_r / env.mdp.delivery_reward)))
        if done:
            break

    return {
        "records": records,
        "sparse_reward": sparse_total,
        "dense_reward": dense_total,
        "success": success,
        "success_steps": success_steps,
        "delivery_count": delivery_count,
        "length": len(records),
        "done": done,
    }


def player_pose(env, player_idx):
    player = env.base_env.state.players[player_idx]
    return player.position, player.orientation


def player_object_name(env, player_idx):
    held_object = env.base_env.state.players[player_idx].held_object
    return None if held_object is None else held_object.name


def pot_object(env, pot_pos):
    return env.base_env.state.objects.get(pot_pos)


def pot_is_empty(env, pot_pos):
    return pot_object(env, pot_pos) is None


def pot_item_count(env, pot_pos):
    obj = pot_object(env, pot_pos)
    if obj is None or obj.name != "soup":
        return 0
    return obj.state[1]


def pot_is_full(env, pot_pos):
    return pot_item_count(env, pot_pos) >= env.mdp.num_items_for_soup


def pot_is_ready(env, pot_pos):
    obj = pot_object(env, pot_pos)
    return (
        obj is not None
        and obj.name == "soup"
        and obj.state[1] == env.mdp.num_items_for_soup
        and obj.state[2] >= env.mdp.soup_cooking_time
    )


class ExpertRollout:
    def __init__(self, env, horizon):
        self.env = env
        self.horizon = horizon
        self.obs = env.multi_reset()
        self.records = []
        self.sparse_reward = 0.0
        self.dense_reward = 0.0
        self.success = 0.0
        self.success_steps = []
        self.delivery_count = 0
        self.done = False

    @property
    def remaining(self):
        return self.horizon - len(self.records)

    def step_joint(self, joint_action):
        if self.done or self.remaining <= 0:
            return False
        action0 = Action.ACTION_TO_INDEX[joint_action[0]]
        action1 = Action.ACTION_TO_INDEX[joint_action[1]]
        self.records.append((self.obs[0], self.obs[1], action0, action1))
        self.obs, rewards, self.done, info = self.env.multi_step(action0, action1)
        self.dense_reward += float(rewards[0])
        sparse_r = float(info.get("sparse_r", 0.0))
        self.sparse_reward += sparse_r
        self.success = max(self.success, float(info.get("success", 0.0)))
        if sparse_r > 0:
            self.success_steps.append(len(self.records) - 1)
            self.delivery_count += max(
                1, int(round(sparse_r / self.env.mdp.delivery_reward)))
        return not self.done

    def run_actor_actions(self, actor_idx, actions):
        for action in actions:
            joint = [Action.STAY, Action.STAY]
            joint[actor_idx] = action
            if not self.step_joint(tuple(joint)):
                return False
        return True

    def run_joint_actions(self, joint_actions):
        for joint_action in joint_actions:
            if not self.step_joint(joint_action):
                return False
        return True

    def run_parallel_actions(self, actions0, actions1):
        max_len = max(len(actions0), len(actions1))
        for idx in range(max_len):
            joint = (
                actions0[idx] if idx < len(actions0) else Action.STAY,
                actions1[idx] if idx < len(actions1) else Action.STAY,
            )
            if not self.step_joint(joint):
                return False
        return True

    def wait(self, steps):
        for _ in range(steps):
            if not self.step_joint((Action.STAY, Action.STAY)):
                return False
        return True

    def wait_until_pot_ready(self, pot_pos):
        while self.remaining > 0 and not self.done and not pot_is_ready(self.env, pot_pos):
            if not self.step_joint((Action.STAY, Action.STAY)):
                return False
        return pot_is_ready(self.env, pot_pos)

    def finish_with_stay(self):
        while self.remaining > 0 and not self.done:
            self.step_joint((Action.STAY, Action.STAY))

    def result(self):
        return {
            "records": self.records,
            "sparse_reward": self.sparse_reward,
            "dense_reward": self.dense_reward,
            "success": self.success,
            "success_steps": self.success_steps,
            "delivery_count": self.delivery_count,
            "length": len(self.records),
            "done": self.done,
        }


def move_actor_to_parking(rollout, actor_idx, parking_pos):
    if parking_pos is None or rollout.remaining <= 0:
        return True
    mdp = rollout.env.mdp
    other_idx = 1 - actor_idx
    pos, orientation = player_pose(rollout.env, actor_idx)
    other_pos, _ = player_pose(rollout.env, other_idx)
    if pos == parking_pos:
        return True
    actions, _, _ = plan_motion_to(
        mdp, pos, orientation, parking_pos, blocked={other_pos})
    return rollout.run_actor_actions(actor_idx, actions)


def finish_fill_from_current_state(rollout, cook_idx, pot_pos):
    mdp = rollout.env.mdp
    while rollout.remaining > 0 and not rollout.done and not pot_is_full(rollout.env, pot_pos):
        cook_pos, cook_orientation = player_pose(rollout.env, cook_idx)
        serve_pos, _ = player_pose(rollout.env, 1 - cook_idx)
        held = player_object_name(rollout.env, cook_idx)
        if held in ("onion", "tomato"):
            actions, _, _ = plan_to_interact(
                mdp, cook_pos, cook_orientation, pot_pos, blocked={serve_pos})
        else:
            remaining_items = mdp.num_items_for_soup - pot_item_count(
                rollout.env, pot_pos)
            actions, _, _ = build_fill_remaining_plan(
                mdp,
                cook_pos,
                cook_orientation,
                pot_pos,
                blocked={serve_pos},
                item_count=remaining_items,
            )
        if not rollout.run_actor_actions(cook_idx, actions):
            return False
    return pot_is_full(rollout.env, pot_pos)


def candidate_from_rollout(result, metadata):
    success_steps = result.get("success_steps", [])
    intervals = [
        success_steps[idx] - success_steps[idx - 1]
        for idx in range(1, len(success_steps))
    ]
    candidate = {
        **result,
        **metadata,
        "joint_actions": [
            (Action.INDEX_TO_ACTION[r[2]], Action.INDEX_TO_ACTION[r[3]])
            for r in result["records"]
        ],
    }
    candidate["first_success_step"] = success_steps[0] if success_steps else None
    candidate["mean_delivery_interval"] = (
        float(np.mean(intervals)) if intervals else None
    )
    candidate["speed_score"] = sum(
        max(result["length"] - step, 0) for step in success_steps)
    return candidate


def rollout_single_pot_loop(env, seed, demo_horizon):
    mdp = env.mdp
    cook_idx = seed["cook_idx"]
    serve_idx = seed["serve_idx"]
    pot_pos = tuple(seed["pot_pos"])
    rollout = ExpertRollout(env, demo_horizon)

    if seed.get("server_parking") is not None:
        move_actor_to_parking(rollout, serve_idx, tuple(seed["server_parking"]))

    while rollout.remaining > 0 and not rollout.done:
        if not pot_is_empty(env, pot_pos):
            if not rollout.wait_until_pot_ready(pot_pos):
                break
            cook_pos, _ = player_pose(env, cook_idx)
            serve_pos, serve_orientation = player_pose(env, serve_idx)
            serve_actions, _, _ = build_serve_plan(
                mdp, serve_pos, serve_orientation, pot_pos, blocked={cook_pos})
            rollout.run_actor_actions(serve_idx, serve_actions)
            continue

        cook_pos, cook_orientation = player_pose(env, cook_idx)
        serve_pos, _ = player_pose(env, serve_idx)
        fill_actions, _, _ = build_fill_plan(
            mdp, cook_pos, cook_orientation, pot_pos, blocked={serve_pos})
        if not rollout.run_actor_actions(cook_idx, fill_actions):
            break

        if seed.get("cook_parking") is not None:
            move_actor_to_parking(rollout, cook_idx, tuple(seed["cook_parking"]))

        if not rollout.wait_until_pot_ready(pot_pos):
            break

        cook_pos, _ = player_pose(env, cook_idx)
        serve_pos, serve_orientation = player_pose(env, serve_idx)
        serve_actions, _, _ = build_serve_plan(
            mdp, serve_pos, serve_orientation, pot_pos, blocked={cook_pos})
        if not rollout.run_actor_actions(serve_idx, serve_actions):
            break

    rollout.finish_with_stay()
    return candidate_from_rollout(
        rollout.result(),
        {
            "cooperative": True,
            "expert_mode": "single_pot_loop",
            "cook_idx": cook_idx,
            "serve_idx": serve_idx,
            "pot_pos": pot_pos,
            "server_parking": seed.get("server_parking"),
            "cook_parking": seed.get("cook_parking"),
            "prefetch_parking": None,
        },
    )


def rollout_prefetch_single_pot_loop(env, seed, demo_horizon,
                                     prefetch_parking, cook_parking):
    mdp = env.mdp
    cook_idx = seed["cook_idx"]
    serve_idx = seed["serve_idx"]
    pot_pos = tuple(seed["pot_pos"])
    rollout = ExpertRollout(env, demo_horizon)

    while rollout.remaining > 0 and not rollout.done:
        if not pot_is_empty(env, pot_pos):
            if not rollout.wait_until_pot_ready(pot_pos):
                break
            cook_pos, _ = player_pose(env, cook_idx)
            serve_pos, serve_orientation = player_pose(env, serve_idx)
            held = player_object_name(env, serve_idx)
            if held == "dish":
                serve_actions, _, _ = build_soup_delivery_plan(
                    mdp, serve_pos, serve_orientation, pot_pos, blocked={cook_pos})
            elif held == "soup":
                serve_actions, _, _ = build_deliver_held_soup_plan(
                    mdp, serve_pos, serve_orientation, blocked={cook_pos})
            else:
                serve_actions, _, _ = build_serve_plan(
                    mdp, serve_pos, serve_orientation, pot_pos, blocked={cook_pos})
            rollout.run_actor_actions(serve_idx, serve_actions)
            continue

        cook_pos, cook_orientation = player_pose(env, cook_idx)
        serve_pos, serve_orientation = player_pose(env, serve_idx)
        fill_actions, _, _ = build_fill_plan(
            mdp, cook_pos, cook_orientation, pot_pos, blocked=set())

        held = player_object_name(env, serve_idx)
        if held == "dish":
            if prefetch_parking is None:
                prefetch_actions = []
            else:
                prefetch_actions, _, _ = plan_motion_to(
                    mdp,
                    serve_pos,
                    serve_orientation,
                    prefetch_parking,
                    blocked=set(),
                )
        elif held is None:
            prefetch_actions, _, _ = build_dish_parking_plan(
                mdp,
                serve_pos,
                serve_orientation,
                prefetch_parking,
                blocked=set(),
            )
        else:
            prefetch_actions = []

        actions0 = fill_actions if cook_idx == 0 else prefetch_actions
        actions1 = prefetch_actions if cook_idx == 0 else fill_actions
        rollout.run_parallel_actions(actions0, actions1)

        if not finish_fill_from_current_state(rollout, cook_idx, pot_pos):
            break

        if player_object_name(env, serve_idx) is None:
            serve_pos, serve_orientation = player_pose(env, serve_idx)
            cook_pos, _ = player_pose(env, cook_idx)
            dish_actions, _, _ = build_dish_parking_plan(
                mdp,
                serve_pos,
                serve_orientation,
                prefetch_parking,
                blocked={cook_pos},
            )
            rollout.run_actor_actions(serve_idx, dish_actions)
        elif player_object_name(env, serve_idx) == "dish" and prefetch_parking is not None:
            move_actor_to_parking(rollout, serve_idx, prefetch_parking)

        if cook_parking is not None:
            move_actor_to_parking(rollout, cook_idx, cook_parking)

        if not rollout.wait_until_pot_ready(pot_pos):
            break

        cook_pos, _ = player_pose(env, cook_idx)
        serve_pos, serve_orientation = player_pose(env, serve_idx)
        held = player_object_name(env, serve_idx)
        if held == "dish":
            serve_actions, _, _ = build_soup_delivery_plan(
                mdp, serve_pos, serve_orientation, pot_pos, blocked={cook_pos})
        elif held == "soup":
            serve_actions, _, _ = build_deliver_held_soup_plan(
                mdp, serve_pos, serve_orientation, blocked={cook_pos})
        else:
            serve_actions, _, _ = build_serve_plan(
                mdp, serve_pos, serve_orientation, pot_pos, blocked={cook_pos})
        if not rollout.run_actor_actions(serve_idx, serve_actions):
            break

    rollout.finish_with_stay()
    return candidate_from_rollout(
        rollout.result(),
        {
            "cooperative": True,
            "expert_mode": "prefetch_single_pot_loop",
            "cook_idx": cook_idx,
            "serve_idx": serve_idx,
            "pot_pos": pot_pos,
            "server_parking": seed.get("server_parking"),
            "cook_parking": cook_parking,
            "prefetch_parking": prefetch_parking,
        },
    )


def rollout_two_pot_pipeline(env, seed, demo_horizon, second_pot,
                             prefetch_parking=None, serve_delay=0):
    mdp = env.mdp
    cook_idx = seed["cook_idx"]
    serve_idx = seed["serve_idx"]
    first_pot = tuple(seed["pot_pos"])
    second_pot = tuple(second_pot)
    rollout = ExpertRollout(env, demo_horizon)

    cook_pos, cook_orientation = player_pose(env, cook_idx)
    serve_pos, serve_orientation = player_pose(env, serve_idx)
    fill_first, _, _ = build_fill_plan(
        mdp, cook_pos, cook_orientation, first_pot, blocked=set())
    if prefetch_parking is None:
        dish_plan = []
    else:
        dish_plan, _, _ = build_dish_parking_plan(
            mdp, serve_pos, serve_orientation, prefetch_parking, blocked=set())
    actions0 = fill_first if cook_idx == 0 else dish_plan
    actions1 = dish_plan if cook_idx == 0 else fill_first
    if not rollout.run_parallel_actions(actions0, actions1):
        rollout.finish_with_stay()
        return candidate_from_rollout(rollout.result(), {
            "cooperative": True,
            "expert_mode": "two_pot_pipeline",
            "cook_idx": cook_idx,
            "serve_idx": serve_idx,
            "pot_pos": first_pot,
            "second_pot_pos": second_pot,
            "server_parking": seed.get("server_parking"),
            "cook_parking": seed.get("cook_parking"),
            "prefetch_parking": prefetch_parking,
            "serve_delay": serve_delay,
        })

    cook_pos, cook_orientation = player_pose(env, cook_idx)
    serve_pos, serve_orientation = player_pose(env, serve_idx)
    fill_second, _, _ = build_fill_plan(
        mdp, cook_pos, cook_orientation, second_pot, blocked=set())
    if player_object_name(env, serve_idx) == "dish":
        serve_first, _, _ = build_soup_delivery_plan(
            mdp, serve_pos, serve_orientation, first_pot, blocked=set())
    else:
        serve_first, _, _ = build_serve_plan(
            mdp, serve_pos, serve_orientation, first_pot, blocked=set())
    serve_first = [Action.STAY] * serve_delay + serve_first
    actions0 = fill_second if cook_idx == 0 else serve_first
    actions1 = serve_first if cook_idx == 0 else fill_second
    rollout.run_parallel_actions(actions0, actions1)

    if not pot_is_ready(env, first_pot) and pot_object(env, first_pot) is not None:
        rollout.wait_until_pot_ready(first_pot)
    if pot_object(env, first_pot) is not None:
        cook_pos, _ = player_pose(env, cook_idx)
        serve_pos, serve_orientation = player_pose(env, serve_idx)
        if player_object_name(env, serve_idx) == "soup":
            delivery_actions, _, _ = build_deliver_held_soup_plan(
                mdp, serve_pos, serve_orientation, blocked={cook_pos})
        elif player_object_name(env, serve_idx) == "dish":
            delivery_actions, _, _ = build_soup_delivery_plan(
                mdp, serve_pos, serve_orientation, first_pot, blocked={cook_pos})
        else:
            delivery_actions, _, _ = build_serve_plan(
                mdp, serve_pos, serve_orientation, first_pot, blocked={cook_pos})
        rollout.run_actor_actions(serve_idx, delivery_actions)

    if pot_object(env, second_pot) is None or (
            pot_object(env, second_pot).state[1] < mdp.num_items_for_soup):
        cook_pos, cook_orientation = player_pose(env, cook_idx)
        serve_pos, _ = player_pose(env, serve_idx)
        try:
            fill_second, _, _ = build_fill_plan(
                mdp, cook_pos, cook_orientation, second_pot, blocked={serve_pos})
            rollout.run_actor_actions(cook_idx, fill_second)
        except RuntimeError:
            pass

    if seed.get("cook_parking") is not None:
        move_actor_to_parking(rollout, cook_idx, tuple(seed["cook_parking"]))

    if rollout.wait_until_pot_ready(second_pot):
        cook_pos, _ = player_pose(env, cook_idx)
        serve_pos, serve_orientation = player_pose(env, serve_idx)
        if player_object_name(env, serve_idx) == "dish":
            serve_second, _, _ = build_soup_delivery_plan(
                mdp, serve_pos, serve_orientation, second_pot, blocked={cook_pos})
        elif player_object_name(env, serve_idx) == "soup":
            serve_second, _, _ = build_deliver_held_soup_plan(
                mdp, serve_pos, serve_orientation, blocked={cook_pos})
        else:
            serve_second, _, _ = build_serve_plan(
                mdp, serve_pos, serve_orientation, second_pot, blocked={cook_pos})
        rollout.run_actor_actions(serve_idx, serve_second)

    rollout.finish_with_stay()
    return candidate_from_rollout(
        rollout.result(),
        {
            "cooperative": True,
            "expert_mode": "two_pot_pipeline",
            "cook_idx": cook_idx,
            "serve_idx": serve_idx,
            "pot_pos": first_pot,
            "second_pot_pos": second_pot,
            "server_parking": seed.get("server_parking"),
            "cook_parking": seed.get("cook_parking"),
            "prefetch_parking": prefetch_parking,
            "serve_delay": serve_delay,
        },
    )


def better_candidate(candidate, best):
    if best is None:
        return True
    candidate_key = (
        candidate["delivery_count"],
        candidate["sparse_reward"],
        candidate.get("speed_score", 0),
        -candidate["success_steps"][0] if candidate["success_steps"] else -10**9,
    )
    best_key = (
        best["delivery_count"],
        best["sparse_reward"],
        best.get("speed_score", 0),
        -best["success_steps"][0] if best["success_steps"] else -10**9,
    )
    return candidate_key > best_key


def nearby_parking_candidates(mdp, target_positions, limit=10):
    candidates = [None]
    for target_pos in target_positions:
        candidates.extend(pos for pos, _ in adjacent_targets(mdp, target_pos))
    candidates.extend(
        sorted(
            mdp.get_valid_player_positions(),
            key=lambda p: min(
                abs(p[0] - target[0]) + abs(p[1] - target[1])
                for target in target_positions),
        )[:limit]
    )
    deduped = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def find_single_delivery_seed(env):
    mdp = env.mdp
    parking_options = [None] + list(mdp.get_valid_player_positions())
    failures = []

    for cook_idx in (0, 1):
        for pot_pos in mdp.terrain_pos_dict["P"]:
            for server_parking in parking_options:
                for cook_parking in parking_options:
                    try:
                        joint_actions = cooperative_joint_actions(
                            mdp, cook_idx, pot_pos,
                            server_parking, cook_parking)
                        result = rollout_joint_actions(env, joint_actions)
                    except RuntimeError as exc:
                        failures.append(str(exc))
                        continue
                    if result["success"] <= 0:
                        continue
                    candidate = {
                        **result,
                        "joint_actions": joint_actions[:result["length"]],
                        "cooperative": True,
                        "cook_idx": cook_idx,
                        "serve_idx": 1 - cook_idx,
                        "pot_pos": pot_pos,
                        "server_parking": server_parking,
                        "cook_parking": cook_parking,
                    }
                    return candidate

    best = None
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
                    "cooperative": False,
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


def find_demonstration(env, demo_horizon, min_deliveries):
    seed = find_single_delivery_seed(env)
    if not seed.get("cooperative", False):
        seed["expert_mode"] = "single_agent"
        return seed

    candidates = []
    try:
        single_pot_candidate = rollout_single_pot_loop(env, seed, demo_horizon)
        candidates.append(single_pot_candidate)
    except RuntimeError:
        pass

    mdp = env.mdp
    prefetch_parking_candidates = nearby_parking_candidates(
        mdp,
        [tuple(seed["pot_pos"])] + list(mdp.terrain_pos_dict["S"]),
        limit=8,
    )
    cook_parking_candidates = nearby_parking_candidates(
        mdp,
        [tuple(seed["pot_pos"])] + list(mdp.terrain_pos_dict["S"]),
        limit=8,
    )
    for prefetch_parking in prefetch_parking_candidates:
        for cook_parking in cook_parking_candidates:
            try:
                candidates.append(rollout_prefetch_single_pot_loop(
                    env,
                    seed,
                    demo_horizon,
                    prefetch_parking=prefetch_parking,
                    cook_parking=cook_parking,
                ))
            except RuntimeError:
                continue

    if len(mdp.terrain_pos_dict["P"]) > 1:
        first_pot = tuple(seed["pot_pos"])
        deduped_parking = nearby_parking_candidates(
            mdp, mdp.terrain_pos_dict["P"], limit=8)
        for second_pot in mdp.terrain_pos_dict["P"]:
            if tuple(second_pot) == first_pot:
                continue
            for prefetch_parking in deduped_parking:
                for serve_delay in (0, 5, 10, 15, 20):
                    try:
                        candidates.append(rollout_two_pot_pipeline(
                            env,
                            seed,
                            demo_horizon,
                            second_pot,
                            prefetch_parking=prefetch_parking,
                            serve_delay=serve_delay,
                        ))
                    except RuntimeError:
                        continue

    best = None
    for candidate in candidates:
        if better_candidate(candidate, best):
            best = candidate

    if best is None:
        raise RuntimeError("Could not build a multi-round plan-BC demonstration")
    if best["delivery_count"] < min_deliveries:
        raise RuntimeError(
            "Plan-BC demonstration did not meet the delivery threshold: "
            f"{best['delivery_count']} < {min_deliveries}. "
            f"Best mode={best.get('expert_mode')} "
            f"sparse={best['sparse_reward']} steps={best['success_steps']}"
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
    parser.add_argument("--demo-horizon", type=int, default=400)
    parser.add_argument("--min-deliveries", type=int, default=1)
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
    demonstration = find_demonstration(
        env, args.demo_horizon, args.min_deliveries)
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
        "demo_horizon": args.demo_horizon,
        "min_deliveries": args.min_deliveries,
        "delivery_count": demonstration.get("delivery_count", 0),
        "success_steps": demonstration.get("success_steps", []),
        "first_success_step": demonstration.get("first_success_step"),
        "mean_delivery_interval": demonstration.get("mean_delivery_interval"),
        "speed_score": demonstration.get("speed_score"),
        "demo_dense_reward": demonstration["dense_reward"],
        "demo_sparse_reward": demonstration["sparse_reward"],
        "expert_mode": demonstration.get("expert_mode"),
        "cooperative": demonstration.get("cooperative", False),
        "active_idx": demonstration.get("active_idx"),
        "cook_idx": demonstration.get("cook_idx"),
        "serve_idx": demonstration.get("serve_idx"),
        "pot_pos": list(demonstration["pot_pos"]),
        "second_pot_pos": (
            list(demonstration["second_pot_pos"])
            if demonstration.get("second_pot_pos") is not None
            else None
        ),
        "parking_pos": (
            list(demonstration["parking_pos"])
            if demonstration["parking_pos"] is not None
            else None
        ) if "parking_pos" in demonstration else None,
        "server_parking": (
            list(demonstration["server_parking"])
            if demonstration.get("server_parking") is not None
            else None
        ),
        "cook_parking": (
            list(demonstration["cook_parking"])
            if demonstration.get("cook_parking") is not None
            else None
        ),
        "prefetch_parking": (
            list(demonstration["prefetch_parking"])
            if demonstration.get("prefetch_parking") is not None
            else None
        ),
        "serve_delay": demonstration.get("serve_delay"),
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
        writer.add_scalar("demo/delivery_count",
                          metadata["delivery_count"], args.bc_epochs)
        writer.close()

    env.close()
    print(
        f"Plan-BC {layout_name}: "
        f"demo_len={demonstration['length']} "
        f"deliveries={demonstration.get('delivery_count', 0)} "
        f"mode={demonstration.get('expert_mode')} "
        f"acc0={metrics['accuracy0']:.3f} "
        f"acc1={metrics['accuracy1']:.3f} "
        f"dense={stats.dense_reward_mean:.3f} "
        f"sparse={stats.sparse_reward_mean:.3f} "
        f"success={stats.success_rate:.3f}"
    )


if __name__ == "__main__":
    main()
