#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-7}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PPO_RESULTS_DIR="${PPO_RESULTS_DIR:-results/PPO}"
ALGO_NAME="${ALGO_NAME:-DQN_improved}"
TIMESTEPS="${TIMESTEPS:-1000000}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
GIF_FPS="${GIF_FPS:-2}"
SEED="${SEED:-10}"
FRAMES="${FRAMES:-1}"
FORCE="${FORCE:-0}"
LAYOUTS="${LAYOUTS:-five_by_five random0 random1 random2 schelling schelling_s unident unident_s scenario2 scenario2_s}"

DQN_LEARNING_RATE="${DQN_LEARNING_RATE:-0.00005}"
DQN_BUFFER_SIZE="${DQN_BUFFER_SIZE:-200000}"
DQN_LEARNING_STARTS="${DQN_LEARNING_STARTS:-20000}"
DQN_BATCH_SIZE="${DQN_BATCH_SIZE:-128}"
DQN_TRAIN_FREQ="${DQN_TRAIN_FREQ:-4}"
DQN_GRADIENT_STEPS="${DQN_GRADIENT_STEPS:-1}"
DQN_TARGET_UPDATE_INTERVAL="${DQN_TARGET_UPDATE_INTERVAL:-10000}"
DQN_EXPLORATION_FRACTION="${DQN_EXPLORATION_FRACTION:-0.6}"
DQN_EXPLORATION_FINAL_EPS="${DQN_EXPLORATION_FINAL_EPS:-0.05}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/luoey/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

json_config() {
    printf '{"layout_name":"%s"}' "$1"
}

dqn_config() {
    printf '{"learning_rate":%s,"buffer_size":%s,"learning_starts":%s,"batch_size":%s,"train_freq":%s,"gradient_steps":%s,"target_update_interval":%s,"exploration_fraction":%s,"exploration_final_eps":%s,"verbose":0}' \
        "${DQN_LEARNING_RATE}" \
        "${DQN_BUFFER_SIZE}" \
        "${DQN_LEARNING_STARTS}" \
        "${DQN_BATCH_SIZE}" \
        "${DQN_TRAIN_FREQ}" \
        "${DQN_GRADIENT_STEPS}" \
        "${DQN_TARGET_UPDATE_INTERVAL}" \
        "${DQN_EXPLORATION_FRACTION}" \
        "${DQN_EXPLORATION_FINAL_EPS}"
}

fixed_ppo_partner_config() {
    printf '{"type":"PPO","location":"%s"}' "$1"
}

model_config() {
    printf '{"type":"%s","location":"%s"}' "$1" "$2"
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
    if [ -f "${base}-best.zip" ]; then
        echo "${base}-best"
    else
        echo "${base}-final"
    fi
}

ppo_partner_base() {
    layout="$1"
    best="${PPO_RESULTS_DIR}/${layout}/models/alt-best"
    final="${PPO_RESULTS_DIR}/${layout}/models/alt-final"
    if [ -f "${best}.zip" ]; then
        echo "${best}"
    elif [ -f "${final}.zip" ]; then
        echo "${final}"
    else
        echo ""
    fi
}

train_one() {
    layout="$1"
    partner_base="$(ppo_partner_base "${layout}")"
    out_dir="$(layout_dir "${layout}")"
    models_dir="${out_dir}/models"
    logs_dir="${out_dir}/logs"
    mkdir -p "${models_dir}" "${logs_dir}"

    if [ -z "${partner_base}" ]; then
        echo "Skipping ${layout}: missing PPO partner under ${PPO_RESULTS_DIR}/${layout}/models"
        return 0
    fi

    ego_base="${models_dir}/ego"
    alt_base="${models_dir}/alt"
    tensorboard_name="${ALGO_NAME}_${layout}_seed${SEED}"

    if [ "${FORCE}" != "1" ] && \
       { [ -f "${ego_base}-best.zip" ] || [ -f "${ego_base}-final.zip" ]; }; then
        echo "==> Skipping ${ALGO_NAME} on ${layout}: model already exists"
        return 0
    fi

    echo "==> Training ${ALGO_NAME} on ${layout} with fixed PPO partner ${partner_base}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" trainer.py \
        OvercookedMultiEnv-v0 DQN FIXED \
        --env-config "$(json_config "${layout}")" \
        --seed "${SEED}" \
        --total-timesteps "${TIMESTEPS}" \
        --framestack "${FRAMES}" \
        --ego-config "$(dqn_config)" \
        --alt-config "$(fixed_ppo_partner_config "${partner_base}")" \
        --tensorboard-log "${logs_dir}" \
        --tensorboard-name "${tensorboard_name}" \
        --eval-freq "${EVAL_FREQ}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --ego-save "${ego_base}-final" \
        --alt-save "${alt_base}-final" \
        --best-ego-save "${ego_base}-best" \
        --best-alt-save "${alt_base}-best"
}

gif_one() {
    layout="$1"
    ego_base="$(best_or_final "$(ego_model_base "${layout}")")"
    alt_base="$(best_or_final "$(alt_model_base "${layout}")")"
    gifs_dir="${RESULTS_DIR}/gifs"
    mkdir -p "${gifs_dir}"

    if [ ! -f "${ego_base}.zip" ] || [ ! -f "${alt_base}.zip" ]; then
        echo "Skipping GIF for ${layout}: missing improved DQN model"
        return 0
    fi

    echo "==> Rendering GIF for ${ALGO_NAME} on ${layout}"
    "${PYTHON_BIN}" render_policy_gif.py OvercookedMultiEnv-v0 \
        --env-config "$(json_config "${layout}")" \
        --ego-config "$(model_config DQN "${ego_base}")" \
        --partner-config "$(model_config PPO "${alt_base}")" \
        --framestack "${FRAMES}" \
        --output "${gifs_dir}/${ALGO_NAME}_${layout}.gif" \
        --fps "${GIF_FPS}" \
        --actions-output "${gifs_dir}/${ALGO_NAME}_${layout}.json"
}

train_all() {
    for layout in ${LAYOUTS}; do
        train_one "${layout}"
    done
}

gifs_all() {
    for layout in ${LAYOUTS}; do
        gif_one "${layout}"
    done
}

latest_all() {
    echo "${ALGO_NAME}"
    for layout in ${LAYOUTS}; do
        ego_base="$(best_or_final "$(ego_model_base "${layout}")")"
        alt_base="$(best_or_final "$(alt_model_base "${layout}")")"
        if [ -f "${ego_base}.zip" ] || [ -f "${alt_base}.zip" ]; then
            echo "  ${layout}: ego=${ego_base}.zip alt=${alt_base}.zip"
        fi
    done
}

case "${1:-help}" in
    train)
        train_all
        ;;
    gifs|gif)
        gifs_all
        ;;
    all)
        train_all
        gifs_all
        ;;
    latest)
        latest_all
        ;;
    help|*)
        echo "Usage: bash run1.sh {train|gifs|all|latest}"
        echo
        echo "Script-only DQN improvement: DQN ego trained against a fixed PPO partner."
        echo
        echo "Examples:"
        echo "  bash run1.sh train"
        echo "  bash run1.sh gifs"
        echo "  LAYOUTS='simple random0' TIMESTEPS=200000 bash run1.sh all"
        echo "  FORCE=1 LAYOUTS='corridor' bash run1.sh train"
        echo "  PYTHON_BIN=/data/luoey/conda_envs/overcooked/bin/python bash run1.sh all"
        ;;
esac
