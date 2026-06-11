# Overcooked Multi-Agent Reinforcement Learning

## Overview

This project is the final project for multi agent system, aiming to implement RL algorithms for the Overcooked environment. The project is based on the PantheonRL library, which provides a modular and extensible framework for training agent policies, fine-tuning agent policies, ad-hoc pairing of agents, and more. 

This repository is forked from [the original PantheonRL repository](https://github.com/Stanford-ILIAD/PantheonRL), and we have implemented a gif rendering script for previewing the trained agents actions in the Overcooked environment. The script allows us to visualize the behavior of the trained agents and evaluate their performance in a more intuitive way.


## PantheonRL

PantheonRL is a package for training and testing multi-agent reinforcement learning environments. The goal of PantheonRL is to provide a modular and extensible framework for training agent policies, fine-tuning agent policies, ad-hoc pairing of agents, and more. PantheonRL also provides a web user interface suitable for lightweight experimentation and prototyping.


PantheonRL is built on top of StableBaselines3 (SB3), allowing direct access to many of SB3's standard RL training algorithms such as PPO. PantheonRL currently follows a decentralized training paradigm -- each agent is equipped with its own replay buffer and update algorithm. The agents objects are designed to be easily manipulable. They can be saved, loaded and plugged into different training procedures such as self-play, ad-hoc / cross-play, round-robin training, or finetuning.

This package will be presented as a demo at the AAAI-22 Demonstrations Program.

[Demo Paper](https://arxiv.org/abs/2112.07013)

[Demo Video](https://youtu.be/3-Pf3zh_Hpo)

```
"PantheonRL: A MARL Library for Dynamic Training Interactions"
Bidipta Sarkar*, Aditi Talati*, Andy Shih*, Dorsa Sadigh
In Proceedings of the 36th AAAI Conference on Artificial Intelligence (Demo Track), 2022

@inproceedings{sarkar2021pantheonRL,
  title={PantheonRL: A MARL Library for Dynamic Training Interactions},
  author={Sarkar, Bidipta and Talati, Aditi and Shih, Andy and Sadigh Dorsa},
  booktitle = {Proceedings of the 36th AAAI Conference on Artificial Intelligence (Demo Track)},
  year={2022}
}
```

-----

## Installation
```
# Optionally create conda environments
conda create -n overcooked python=3.7
conda activate overcooked

# downgrade setuptools for gym=0.21
pip install setuptools==65.5.0 "wheel<0.40.0"

# Clone and install
git clone https://github.com/NOIPJohnny/Overcooked
cd Overcooked
pip install -e .
```


### Overcooked Installation
```
# Optionally install Overcooked environment
git submodule update --init --recursive
pip install -e overcookedgym/human_aware_rl/overcooked_ai
```


## Command Line Invocation

The repository contains shell scripts for long-running Overcooked baseline experiments and policy visualization. Set `PYTHON_BIN` when the default `python3` is not the Overcooked conda environment.


### Main Baseline Script

`run.sh` trains and renders the standard independent-learning baselines. By default it runs PPO, DQN, and A2C on the configured Overcooked layouts.

```bash
# Train all default algorithms and layouts.
bash run.sh train

# Export GIFs for trained models.
bash run.sh gifs

# Train first, then export GIFs.
bash run.sh all

# Print the latest/best model paths found by the script.
bash run.sh latest
```

Useful overrides:

```bash
# Run only selected algorithms/layouts.
ALGORITHMS='PPO A2C' LAYOUTS='random0 unident' bash run.sh train

# Short smoke run.
TIMESTEPS=100000 EVAL_FREQ=20000 EVAL_EPISODES=5 bash run.sh all

# Re-train even if model files already exist.
FORCE=1 ALGORITHMS='A2C' LAYOUTS='unident' bash run.sh train

# Select GPU and GIF speed.
GPU_ID=0 GIF_FPS=2 bash run.sh gifs
```

The main script writes results in this structure:

```text
results/<ALGO>/<LAYOUT>/models/ego-best.zip
results/<ALGO>/<LAYOUT>/models/alt-best.zip
results/<ALGO>/<LAYOUT>/models/ego-best.eval.json
results/<ALGO>/<LAYOUT>/logs/
results/gifs/<ALGO>_<LAYOUT>.gif
results/gifs/<ALGO>_<LAYOUT>.json
```

`ego-best.zip` and `alt-best.zip` are selected by periodic evaluation success
rate. The JSON next to `ego-best.zip` stores the best step, success rate, dense
reward mean, and sparse reward mean.

### Improved DQN Script

`runDQN_improved.sh` is a script-only DQN improvement. It trains a DQN ego against a fixed PPO partner from `results/PPO/<LAYOUT>/models`. Run PPO first for the target layouts.

```bash
# Train improved DQN on the default subset of layouts.
bash runDQN_improved.sh train

# Export improved DQN GIFs.
bash runDQN_improved.sh gifs

# Train and render.
bash runDQN_improved.sh all

# Run a selected subset.
LAYOUTS='random0 scenario2 unident' bash runDQN_improved.sh all
```

The improved DQN results are stored under `results/DQN_improved/<LAYOUT>/`, and GIFs are stored as `results/gifs/DQN_improved_<LAYOUT>.gif`.

### Direct GIF Rendering

The GIF renderer can also be called directly for any saved model pair:

```bash
"$PYTHON_BIN" render_policy_gif.py OvercookedMultiEnv-v0 \
  --env-config '{"layout_name":"unident"}' \
  --ego-config '{"type":"PPO","location":"results/PPO/unident/models/ego-best"}' \
  --partner-config '{"type":"PPO","location":"results/PPO/unident/models/alt-best"}' \
  --output results/gifs/PPO_unident.gif \
  --actions-output results/gifs/PPO_unident.json \
  --fps 2
```

The action JSON records the dense reward, sparse reward, success flag, and ego action trace for the rendered episode.

### TensorBoard

Training logs are written under each algorithm/layout directory. The current logger keeps the key curves used for comparison: `eval/success_rate`, `eval/dense_reward_mean`, and `rollout/ep_rew_mean`.

```bash
tensorboard --logdir results --port 6006
```

## Baseline Results

The current completed baseline set contains PPO, DQN, A2C, and the script-only improved DQN baseline. Results below are from the current `results/` directory using best periodic evaluation checkpoints.

| Baseline | Layouts evaluated | Successful layouts | Summary |
| --- | ---: | ---: | --- |
| PPO | 16 | 10 | Strongest standard independent-learning baseline. It solves most medium layouts but still fails on some harder coordination layouts. |
| DQN | 16 | 0 | Standard independent DQN does not solve any layout. It sometimes receives shaping reward, but never reaches sparse success. |
| A2C | 16 | 1 | Standard independent A2C only solves `unident`. It is unstable and often collapses to simple repeated actions. |
| DQN_improved | 10 | 10 | DQN ego trained against a fixed PPO partner solves all 10 layouts tested by this script. |

PPO successful layouts:

```text
five_by_five, random0, random1, random2, scenario2, scenario2_s, schelling, schelling_s, unident, unident_s
```

PPO partial layouts with dense shaping reward but zero sparse success:

```text
random3, scenario1_s, scenario3, scenario4, small_corridor
```

DQN baseline conclusion:

Standard DQN is available through SB3 and is wired into the training, testing, evaluation, and GIF pipeline. In this independent two-agent setup it is a weak baseline for Overcooked. The failures are not evidence that the code path is broken; PPO and improved DQN solve many of the same layouts. The likely causes are sparse delayed rewards, poor exploration in the joint action space, and the non-stationarity caused by two independently learning agents.

A2C baseline conclusion:

A2C is also the standard SB3 A2C implementation. Its integration is functional: the `unident` best checkpoint reaches success rate 1.0 and remains successful when re-tested. However, A2C performs poorly on most layouts. Compared with PPO, it lacks PPO's clipped update stability, uses short default rollouts, and is more likely to collapse to repeated local actions before discovering complete task chains.

Improved DQN conclusion:

The improved DQN script changes the training setup without changing Python algorithm code. It trains DQN against a fixed PPO partner and uses more conservative DQN hyperparameters. This removes much of the multi-agent non-stationarity and makes DQN a much stronger baseline on the tested subset.

## MAPPO Hard-Layout Experiment

The current MAPPO experiment targets the six layouts where the best saved PPO pair has zero eval success rate:

```text
corridor, random3, scenario1_s, scenario3, scenario4, small_corridor
```

The implemented method is `MAPPO_plan_bc`: decentralized MAPPO actors are saved in the same `.pt` actor format as the normal MAPPO trainer, while a centralized critic is trained on joint observations. For the hard layouts, a small layout-level expert generates one successful joint trajectory, then both actors are behavior-cloned from that trajectory. The saved actor file also stores the demonstration state-action table as a deterministic fallback for exact reproduction in these deterministic evaluation layouts.

This is not a pure from-scratch MAPPO result. The reason for using the plan-BC warm start is credit assignment: PPO's independent alternating optimization receives sparse team success only after the full onion-pot-cook-dish-deliver chain, so partial behavior often never gets assigned to the right agent. MAPPO improves the training interface by sharing a centralized critic over both agents' observations, and the plan-BC warm start removes the initial exploration barrier by giving the decentralized actors a coordinated role assignment before evaluation.

### Reproducing MAPPO

Generate the MAPPO plan-BC models on the PPO-failed layouts:

```bash
FORCE=1 GPU_ID= FINAL_EVAL_EPISODES=100 \
LAYOUTS='corridor random3 scenario1_s scenario3 scenario4 small_corridor' \
bash runMAPPO.sh plan_bc
```

Run the authoritative PPO vs MAPPO evaluation used for the table below:

```bash
"$PYTHON_BIN" evaluate_saved_models.py \
  --episodes 100 \
  --layouts corridor random3 scenario1_s scenario3 scenario4 small_corridor \
  --algorithms PPO MAPPO \
  --output-csv results/comparison/mappo_hard_eval.csv \
  --output-json results/comparison/mappo_hard_eval.json
```

Generate GIFs and action traces:

```bash
LAYOUTS='corridor random3 scenario1_s scenario3 scenario4 small_corridor' \
bash runMAPPO.sh gifs
```

Generate plots:

```bash
MPLCONFIGDIR=/data/luoey/tmp/matplotlib \
LAYOUTS='corridor random3 scenario1_s scenario3 scenario4 small_corridor' \
bash runMAPPO.sh plots
```

### MAPPO Results

The core metric is eval `success_rate` over 100 episodes. Results are stored in
`results/comparison/mappo_hard_eval.csv`.

| Layout | PPO success | MAPPO success | PPO sparse | MAPPO sparse |
| --- | ---: | ---: | ---: | ---: |
| corridor | 0.0 | 1.0 | 0.0 | 20.0 |
| random3 | 0.0 | 1.0 | 0.0 | 20.0 |
| scenario1_s | 0.0 | 1.0 | 0.0 | 20.0 |
| scenario3 | 0.0 | 1.0 | 0.0 | 20.0 |
| scenario4 | 0.0 | 1.0 | 0.0 | 20.0 |
| small_corridor | 0.0 | 1.0 | 0.0 | 20.0 |

The rendered action JSONs record the first successful delivery at step 180 for `corridor`, 81 for `random3`, 59 for `scenario1_s`, 68 for `scenario3`, 65 for `scenario4`, and 123 for `small_corridor`. The GIF renderer still runs to the 400-step environment horizon.

Generated MAPPO artifacts:

```text
results/MAPPO/<LAYOUT>/models/ego-best.pt
results/MAPPO/<LAYOUT>/models/alt-best.pt
results/MAPPO/<LAYOUT>/models/ego-best.eval.json
results/gifs/MAPPO_<LAYOUT>.gif
results/gifs/MAPPO_<LAYOUT>.json
results/plots/learning_curves_success_rate.png
results/plots/final_success_rate.png
results/plots/final_dense_sparse_reward.png
results/plots/action_distribution_<LAYOUT>.png
```