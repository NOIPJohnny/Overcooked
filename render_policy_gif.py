import argparse
import json
import os

from PIL import Image

from overcooked_ai_py.mdp.actions import Action

from pantheonrl.common.evaluation import load_policy, make_eval_env, static_agent


def render_rgb(env):
    current = env
    while hasattr(current, "env"):
        try:
            return current.render(mode="rgb_array")
        except (NotImplementedError, AttributeError, ValueError):
            current = current.env
    return current.render(mode="rgb_array")


def action_name(action):
    action_tuple = Action.INDEX_TO_ACTION[int(action)]
    return Action.ACTION_TO_CHAR[action_tuple]


def observation_shape(policy_or_model):
    if hasattr(policy_or_model, "policy"):
        return policy_or_model.policy.observation_space.shape
    return policy_or_model.observation_space.shape


def validate_observation_shape(name, policy_or_model, config, env, env_config):
    expected = observation_shape(policy_or_model)
    actual = env.observation_space.shape
    if expected != actual:
        raise ValueError(
            f"{name} observation shape mismatch: model at "
            f"{config['location']} expects {expected}, but env "
            f"{config['type']} with {env_config} provides {actual}."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Render a saved Overcooked policy pair to a GIF.")
    parser.add_argument("env", help="Gym environment id")
    parser.add_argument("--env-config", type=json.loads, required=True)
    parser.add_argument("--ego-config", type=json.loads, required=True)
    parser.add_argument("--partner-config", type=json.loads, required=True)
    parser.add_argument("--framestack", type=int, default=1)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", default="policy_rollout.gif")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--actions-output")
    args = parser.parse_args()

    ego_policy = load_policy(
        args.ego_config["type"], args.ego_config["location"], args.ego_config)
    partner_policy = load_policy(
        args.partner_config["type"],
        args.partner_config["location"],
        args.partner_config,
    )

    env = make_eval_env(args.env, args.env_config, args.framestack)
    validate_observation_shape("ego", ego_policy, args.ego_config, env, args.env_config)
    validate_observation_shape("partner", partner_policy, args.partner_config, env, args.env_config)

    ego = static_agent(ego_policy)
    partner = static_agent(partner_policy)
    env.add_partner_agent(partner)

    frames = []
    trace = []
    obs = None
    for _ in range(args.episode + 1):
        obs = env.reset()
    done = False
    dense_reward = 0.0
    sparse_reward = 0.0
    frames.append(Image.fromarray(render_rgb(env)))

    while not done:
        action = ego.get_action(obs, False)
        obs, reward, done, info = env.step(action)
        dense_reward += reward
        sparse_reward += info.get("sparse_r", 0.0)
        trace.append({
            "step": len(trace),
            "ego_action": int(action),
            "ego_action_name": action_name(action),
            "dense_reward": float(reward),
            "sparse_reward": float(info.get("sparse_r", 0.0)),
            "success": float(info.get("success", 0.0)),
        })
        frames.append(Image.fromarray(render_rgb(env)))

    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    duration = int(1000 / args.fps)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )

    if args.actions_output:
        if os.path.dirname(args.actions_output):
            os.makedirs(os.path.dirname(args.actions_output), exist_ok=True)
        with open(args.actions_output, "w") as f:
            json.dump({
                "dense_reward": dense_reward,
                "sparse_reward": sparse_reward,
                "success": float(sparse_reward > 0),
                "steps": trace,
            }, f, indent=2)

    env.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
