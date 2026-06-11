#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/luoey/conda_envs/overcooked/bin/python}"
GPU_ID="${GPU_ID:-7}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PPO_RESULTS_DIR="${PPO_RESULTS_DIR:-results/PPO}"
ALGO_NAME="${ALGO_NAME:-MAPPO}"
TIMESTEPS="${TIMESTEPS:-500000}"
N_STEPS="${N_STEPS:-2048}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-100}"
GIF_FPS="${GIF_FPS:-2}"
SEED="${SEED:-10}"
FORCE="${FORCE:-0}"
PLAN_BC_MIN_DELIVERIES="${PLAN_BC_MIN_DELIVERIES:-1}"
PLAN_BC_DEMO_HORIZON="${PLAN_BC_DEMO_HORIZON:-400}"
LAYOUTS="${LAYOUTS:-corridor five_by_five random0 random1 random2 random3 scenario1_s scenario2 scenario2_s scenario3 scenario4 schelling schelling_s small_corridor unident unident_s}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/luoey/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

json_config() {
    printf '{"layout_name":"%s"}' "$1"
}

model_config() {
    printf '{"type":"MAPPO","location":"%s"}' "$1"
}

layout_dir() {
    echo "${RESULTS_DIR}/${ALGO_NAME}/$1"
}

ego_model_base() {
    echo "$(layout_dir "$1")/models/ego"
}

alt_model_base() {
    echo "$(layout_dir "$1")/models/alt"
}

best_or_final() {
    base="$1"
    if [ -f "${base}-best.pt" ]; then
        echo "${base}-best"
    else
        echo "${base}-final"
    fi
}

ppo_best_or_final() {
    layout="$1"
    role="$2"
    best="${PPO_RESULTS_DIR}/${layout}/models/${role}-best"
    final="${PPO_RESULTS_DIR}/${layout}/models/${role}-final"
    if [ -f "${best}.zip" ]; then
        echo "${best}"
    elif [ -f "${final}.zip" ]; then
        echo "${final}"
    else
        echo ""
    fi
}

warmstart_one() {
    layout="$1"
    ego_ppo="$(ppo_best_or_final "${layout}" ego)"
    alt_ppo="$(ppo_best_or_final "${layout}" alt)"
    out_dir="$(layout_dir "${layout}")"
    models_dir="${out_dir}/models"
    logs_dir="${out_dir}/logs"
    mkdir -p "${models_dir}" "${logs_dir}"

    if [ -z "${ego_ppo}" ] || [ -z "${alt_ppo}" ]; then
        echo "Skipping warm-start for ${layout}: missing PPO model"
        return 0
    fi

    ego_base="${models_dir}/ego"
    alt_base="${models_dir}/alt"
    checkpoint_base="${models_dir}/checkpoint"
    tensorboard_name="${ALGO_NAME}_${layout}_warmstart_seed${SEED}"

    if [ "${FORCE}" != "1" ] && \
       [ -f "${ego_base}-best.pt" ] && \
       [ -f "${alt_base}-best.pt" ]; then
        echo "==> Skipping warm-start ${ALGO_NAME} on ${layout}: model already exists"
        return 0
    fi

    echo "==> Warm-starting ${ALGO_NAME} on ${layout} from PPO"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" mappo_warmstart.py \
        OvercookedMultiEnv-v0 \
        --env-config "$(json_config "${layout}")" \
        --seed "${SEED}" \
        --ego-ppo "${ego_ppo}" \
        --alt-ppo "${alt_ppo}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --tensorboard-log "${logs_dir}" \
        --tensorboard-name "${tensorboard_name}" \
        --ego-save "${ego_base}-best.pt" \
        --alt-save "${alt_base}-best.pt" \
        --checkpoint-save "${checkpoint_base}-best.pt"
}

train_one() {
    layout="$1"
    out_dir="$(layout_dir "${layout}")"
    models_dir="${out_dir}/models"
    logs_dir="${out_dir}/logs"
    mkdir -p "${models_dir}" "${logs_dir}"

    ego_base="${models_dir}/ego"
    alt_base="${models_dir}/alt"
    checkpoint_base="${models_dir}/checkpoint"
    tensorboard_name="${ALGO_NAME}_${layout}_seed${SEED}"

    if [ "${FORCE}" != "1" ] && \
       { [ -f "${ego_base}-best.pt" ] || [ -f "${ego_base}-final.pt" ]; } && \
       { [ -f "${alt_base}-best.pt" ] || [ -f "${alt_base}-final.pt" ]; }; then
        echo "==> Skipping ${ALGO_NAME} on ${layout}: model already exists"
        return 0
    fi

    echo "==> Training ${ALGO_NAME} on ${layout}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" mappo_trainer.py \
        OvercookedMultiEnv-v0 \
        --env-config "$(json_config "${layout}")" \
        --seed "${SEED}" \
        --total-timesteps "${TIMESTEPS}" \
        --n-steps "${N_STEPS}" \
        --tensorboard-log "${logs_dir}" \
        --tensorboard-name "${tensorboard_name}" \
        --eval-freq "${EVAL_FREQ}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --ego-save "${ego_base}-final.pt" \
        --alt-save "${alt_base}-final.pt" \
        --checkpoint-save "${checkpoint_base}-final.pt" \
        --best-ego-save "${ego_base}-best.pt" \
        --best-alt-save "${alt_base}-best.pt" \
        --best-checkpoint-save "${checkpoint_base}-best.pt"
}

plan_bc_one() {
    layout="$1"
    out_dir="$(layout_dir "${layout}")"
    models_dir="${out_dir}/models"
    logs_dir="${out_dir}/logs"
    mkdir -p "${models_dir}" "${logs_dir}"

    ego_base="${models_dir}/ego"
    alt_base="${models_dir}/alt"
    checkpoint_base="${models_dir}/checkpoint"
    tensorboard_name="${ALGO_NAME}_${layout}_plan_bc_seed${SEED}"

    if [ "${FORCE}" != "1" ] && \
       [ -f "${ego_base}-best.pt" ] && \
       [ -f "${alt_base}-best.pt" ]; then
        echo "==> Skipping plan-BC ${ALGO_NAME} on ${layout}: model already exists"
        return 0
    fi

    echo "==> Plan-BC distilling ${ALGO_NAME} on ${layout}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" mappo_plan_bc.py \
        OvercookedMultiEnv-v0 \
        --env-config "$(json_config "${layout}")" \
        --seed "${SEED}" \
        --eval-episodes "${FINAL_EVAL_EPISODES}" \
        --min-deliveries "${PLAN_BC_MIN_DELIVERIES}" \
        --demo-horizon "${PLAN_BC_DEMO_HORIZON}" \
        --tensorboard-log "${logs_dir}" \
        --tensorboard-name "${tensorboard_name}" \
        --ego-save "${ego_base}-best.pt" \
        --alt-save "${alt_base}-best.pt" \
        --checkpoint-save "${checkpoint_base}-best.pt"
}

gif_one() {
    layout="$1"
    ego_base="$(best_or_final "$(ego_model_base "${layout}")")"
    alt_base="$(best_or_final "$(alt_model_base "${layout}")")"
    gifs_dir="${RESULTS_DIR}/gifs"
    mkdir -p "${gifs_dir}"

    if [ ! -f "${ego_base}.pt" ] || [ ! -f "${alt_base}.pt" ]; then
        echo "Skipping GIF for ${ALGO_NAME}/${layout}: missing model"
        return 0
    fi

    echo "==> Rendering GIF for ${ALGO_NAME} on ${layout}"
    "${PYTHON_BIN}" render_policy_gif.py OvercookedMultiEnv-v0 \
        --env-config "$(json_config "${layout}")" \
        --ego-config "$(model_config "${ego_base}.pt")" \
        --partner-config "$(model_config "${alt_base}.pt")" \
        --output "${gifs_dir}/${ALGO_NAME}_${layout}.gif" \
        --fps "${GIF_FPS}" \
        --actions-output "${gifs_dir}/${ALGO_NAME}_${layout}.json"
}

train_all() {
    for layout in ${LAYOUTS}; do
        train_one "${layout}"
    done
}

plan_bc_all() {
    for layout in ${LAYOUTS}; do
        plan_bc_one "${layout}"
    done
}

warmstart_all() {
    for layout in ${LAYOUTS}; do
        warmstart_one "${layout}"
    done
}

gifs_all() {
    for layout in ${LAYOUTS}; do
        gif_one "${layout}"
    done
}

eval_all() {
    "${PYTHON_BIN}" evaluate_saved_models.py \
        --results-dir "${RESULTS_DIR}" \
        --episodes "${FINAL_EVAL_EPISODES}" \
        --layouts ${LAYOUTS} \
        --algorithms PPO DQN A2C DQN_improved "${ALGO_NAME}" \
        --output-csv "${RESULTS_DIR}/comparison/final_eval.csv" \
        --output-json "${RESULTS_DIR}/comparison/final_eval.json"
}

plots_all() {
    "${PYTHON_BIN}" plot_results.py \
        --results-dir "${RESULTS_DIR}" \
        --layouts ${LAYOUTS} \
        --algorithms PPO DQN A2C DQN_improved "${ALGO_NAME}" \
        --output-dir "${RESULTS_DIR}/plots"
}

latest_all() {
    echo "${ALGO_NAME}"
    for layout in ${LAYOUTS}; do
        ego_base="$(best_or_final "$(ego_model_base "${layout}")")"
        alt_base="$(best_or_final "$(alt_model_base "${layout}")")"
        if [ -f "${ego_base}.pt" ] || [ -f "${alt_base}.pt" ]; then
            echo "  ${layout}: ego=${ego_base}.pt alt=${alt_base}.pt"
        fi
    done
}

case "${1:-help}" in
    warmstart)
        warmstart_all
        ;;
    train)
        train_all
        ;;
    plan_bc)
        plan_bc_all
        ;;
    gifs|gif)
        gifs_all
        ;;
    eval)
        eval_all
        ;;
    plots|plot)
        plots_all
        ;;
    all)
        warmstart_all
        train_all
        gifs_all
        eval_all
        plots_all
        ;;
    latest)
        latest_all
        ;;
    help|*)
        echo "Usage: bash runMAPPO.sh {warmstart|train|plan_bc|gifs|eval|plots|all|latest}"
        echo
        echo "Examples:"
        echo "  bash runMAPPO.sh warmstart"
        echo "  bash runMAPPO.sh train"
        echo "  LAYOUTS='corridor random3 scenario1_s scenario3 scenario4 small_corridor' bash runMAPPO.sh plan_bc"
        echo "  PLAN_BC_MIN_DELIVERIES=2 LAYOUTS='corridor random3 scenario1_s scenario3 scenario4 small_corridor' bash runMAPPO.sh plan_bc"
        echo "  TIMESTEPS=20000 EVAL_FREQ=10000 LAYOUTS='unident' bash runMAPPO.sh all"
        echo "  FORCE=1 LAYOUTS='random3 scenario4' bash runMAPPO.sh train"
        ;;
esac
