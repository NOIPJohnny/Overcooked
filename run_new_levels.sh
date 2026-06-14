#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-7}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SEED="${SEED:-10}"
NEW_LEVEL_EVAL_EPISODES="${NEW_LEVEL_EVAL_EPISODES:-100}"
GIF_FPS="${GIF_FPS:-2}"
LAYOUTS="${LAYOUTS:-chicane_bottleneck_hard asymmetric_corridor_hard double_pot_maze_hard}"
BASELINE_ALGORITHMS="${BASELINE_ALGORITHMS:-PPO DQN A2C}"
MAPPO_ALGO="${MAPPO_ALGO:-MAPPO}"
SMOKE_DIR="${SMOKE_DIR:-/tmp/overcooked_new_levels_smoke}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/luoey/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

json_config() {
    printf '{"layout_name":"%s"}' "$1"
}

run_baselines() {
    PYTHON_BIN="${PYTHON_BIN}" \
    GPU_ID="${GPU_ID}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    SEED="${SEED}" \
    LAYOUTS="${LAYOUTS}" \
    ALGORITHMS="${BASELINE_ALGORITHMS}" \
    bash run.sh train
}

run_mappo() {
    PYTHON_BIN="${PYTHON_BIN}" \
    GPU_ID="${GPU_ID}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    SEED="${SEED}" \
    LAYOUTS="${LAYOUTS}" \
    ALGO_NAME="${MAPPO_ALGO}" \
    bash runMAPPO.sh plan_bc
}

render_gifs() {
    PYTHON_BIN="${PYTHON_BIN}" \
    GPU_ID="${GPU_ID}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    SEED="${SEED}" \
    GIF_FPS="${GIF_FPS}" \
    LAYOUTS="${LAYOUTS}" \
    ALGORITHMS="${BASELINE_ALGORITHMS}" \
    bash run.sh gifs

    PYTHON_BIN="${PYTHON_BIN}" \
    GPU_ID="${GPU_ID}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    SEED="${SEED}" \
    GIF_FPS="${GIF_FPS}" \
    LAYOUTS="${LAYOUTS}" \
    ALGO_NAME="${MAPPO_ALGO}" \
    bash runMAPPO.sh gifs
}

evaluate_all() {
    "${PYTHON_BIN}" evaluate_saved_models.py \
        --results-dir "${RESULTS_DIR}" \
        --episodes "${NEW_LEVEL_EVAL_EPISODES}" \
        --layouts ${LAYOUTS} \
        --algorithms ${BASELINE_ALGORITHMS} "${MAPPO_ALGO}" \
        --output-csv "${RESULTS_DIR}/comparison/new_levels_eval.csv" \
        --output-json "${RESULTS_DIR}/comparison/new_levels_eval.json"
}

plot_all() {
    "${PYTHON_BIN}" plot_results.py \
        --results-dir "${RESULTS_DIR}" \
        --layouts ${LAYOUTS} \
        --algorithms ${BASELINE_ALGORITHMS} "${MAPPO_ALGO}" \
        --output-dir "${RESULTS_DIR}/plots/new_levels"
}

smoke() {
    mkdir -p "${SMOKE_DIR}"
    for layout in ${LAYOUTS}; do
        out_dir="${SMOKE_DIR}/${layout}"
        mkdir -p "${out_dir}"
        "${PYTHON_BIN}" mappo_plan_bc.py \
            OvercookedMultiEnv-v0 \
            --env-config "$(json_config "${layout}")" \
            --seed "${SEED}" \
            --bc-epochs 1 \
            --eval-episodes 1 \
            --demo-horizon 400 \
            --min-deliveries 1 \
            --ego-save "${out_dir}/ego-smoke.pt" \
            --alt-save "${out_dir}/alt-smoke.pt" \
            --checkpoint-save "${out_dir}/checkpoint-smoke.pt"
    done
}

latest() {
    PYTHON_BIN="${PYTHON_BIN}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    LAYOUTS="${LAYOUTS}" \
    ALGORITHMS="${BASELINE_ALGORITHMS}" \
    bash run.sh latest

    PYTHON_BIN="${PYTHON_BIN}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    LAYOUTS="${LAYOUTS}" \
    ALGO_NAME="${MAPPO_ALGO}" \
    bash runMAPPO.sh latest
}

case "${1:-help}" in
    train_baselines)
        run_baselines
        ;;
    train_mappo)
        run_mappo
        ;;
    train)
        run_baselines
        run_mappo
        ;;
    gifs|gif)
        render_gifs
        ;;
    eval)
        evaluate_all
        ;;
    plots|plot)
        plot_all
        ;;
    smoke)
        smoke
        ;;
    latest)
        latest
        ;;
    all)
        run_baselines
        run_mappo
        render_gifs
        evaluate_all
        plot_all
        ;;
    help|*)
        echo "Usage: bash run_new_levels.sh {smoke|train_baselines|train_mappo|train|gifs|eval|plots|latest|all}"
        echo
        echo "Examples:"
        echo "  PYTHON_BIN=/data/luoey/conda_envs/overcooked/bin/python bash run_new_levels.sh smoke"
        echo "  PYTHON_BIN=/data/luoey/conda_envs/overcooked/bin/python bash run_new_levels.sh train_mappo"
        echo "  TIMESTEPS=100000 EVAL_FREQ=20000 bash run_new_levels.sh train_baselines"
        echo "  bash run_new_levels.sh all"
        ;;
esac