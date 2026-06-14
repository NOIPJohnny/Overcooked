import gym
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner, NO_COUNTERS_PARAMS

from overcookedgym.custom_layouts import CUSTOM_LAYOUTS, custom_layout_params
from pantheonrl.common.multiagentenv import SimultaneousEnv


def build_overcooked_mdp(layout_name, rew_shaping_params):
    if layout_name not in CUSTOM_LAYOUTS:
        return OvercookedGridworld.from_layout_name(
            layout_name=layout_name,
            rew_shaping_params=rew_shaping_params)

    layout_params = custom_layout_params(layout_name)
    grid = layout_params.pop("grid")
    return OvercookedGridworld.from_grid(
        grid,
        layout_params,
        {"rew_shaping_params": rew_shaping_params})


class OvercookedMultiEnv(SimultaneousEnv):
    def __init__(self, layout_name, ego_agent_idx=0, baselines=False):
        """
        base_env: OvercookedEnv
        featurize_fn: what function is used to featurize states returned in the 'both_agent_obs' field
        """
        super(OvercookedMultiEnv, self).__init__()

        DEFAULT_ENV_PARAMS = {
            "horizon": 400
        }
        rew_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5,
            "DISH_DISP_DISTANCE_REW": 0,
            "POT_DISTANCE_REW": 0,
            "SOUP_DISTANCE_REW": 0,
        }

        self.mdp = build_overcooked_mdp(layout_name, rew_shaping_params)
        if layout_name in CUSTOM_LAYOUTS:
            mlp = MediumLevelPlanner(self.mdp, NO_COUNTERS_PARAMS)
        else:
            mlp = MediumLevelPlanner.from_pickle_or_compute(self.mdp, NO_COUNTERS_PARAMS, force_compute=False)

        self.base_env = OvercookedEnv(self.mdp, **DEFAULT_ENV_PARAMS)
        self.featurize_fn = lambda x: self.mdp.featurize_state(x, mlp)
        self.last_info = {}

        if baselines: np.random.seed(0)

        self.observation_space = self._setup_observation_space()
        self.lA = len(Action.ALL_ACTIONS)
        self.action_space  = gym.spaces.Discrete( self.lA )
        self.ego_agent_idx = ego_agent_idx
        self.multi_reset()

    def _setup_observation_space(self):
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * np.inf  # max(self.mdp.soup_cooking_time, self.mdp.num_items_for_soup, 5)

        return gym.spaces.Box(-high, high, dtype=np.float64)

    def multi_step(self, ego_action, alt_action):
        """
        action:
            (agent with index self.agent_idx action, other agent action)
            is a tuple with the joint action of the primary and secondary agents in index format
            encoded as an int

        returns:
            observation: formatted to be standard input for self.agent_idx's policy
        """
        ego_action, alt_action = Action.INDEX_TO_ACTION[ego_action], Action.INDEX_TO_ACTION[alt_action]
        if self.ego_agent_idx == 0:
            joint_action = (ego_action, alt_action)
        else:
            joint_action = (alt_action, ego_action)

        next_state, sparse_reward, done, info = self.base_env.step(joint_action)

        # reward shaping
        rew_shape = info['shaped_r']
        reward = sparse_reward + rew_shape
        info = {
            **info,
            "sparse_r": sparse_reward,
            "dense_r": reward,
            "success": float(sparse_reward > 0),
        }
        if done and "episode" in info:
            ep_sparse_r = info["episode"]["ep_sparse_r"]
            info["episode"]["ep_dense_r"] = (
                ep_sparse_r + info["episode"]["ep_shaped_r"])
            info["episode"]["success"] = float(ep_sparse_r > 0)
        self.last_info = info

        #print(self.base_env.mdp.state_string(next_state))
        ob_p0, ob_p1 = self.featurize_fn(next_state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs), (reward, reward), done, info

    def multi_reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        self.last_info = {}
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)
        if self.ego_agent_idx == 0:
            ego_obs, alt_obs = ob_p0, ob_p1
        else:
            ego_obs, alt_obs = ob_p1, ob_p0

        return (ego_obs, alt_obs)

    def render(self, mode='human', close=False):
        if close:
            return None
        if mode == 'rgb_array':
            from pantheonrl.common.overcooked_render import render_state
            return render_state(self.mdp, self.base_env.state)
        if mode == 'human':
            print(self.base_env.mdp.state_string(self.base_env.state))
            return None
        raise ValueError(f"Unsupported render mode: {mode}")
