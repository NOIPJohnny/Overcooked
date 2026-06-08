#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-7}"
RESULTS_DIR="${RESULTS_DIR:-results}"
TIMESTEPS="${TIMESTEPS:-500000}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
GIF_FPS="${GIF_FPS:-2}"
SEED="${SEED:-10}"
FRAMES="${FRAMES:-1}"
FORCE="${FORCE:-0}"
ALGORITHMS="${ALGORITHMS:-PPO DQN A2C}"
LAYOUTS="${LAYOUTS:-corridor five_by_five random0 random1 random2 random3 scenario1_s scenario2 scenario2_s scenario3 scenario4 schelling schelling_s small_corridor unident unident_s}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/data/luoey/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

json_config() {
    printf '{"layout_name":"%s"}' "$1"
}

model_config() {
    printf '{"type":"%s","location":"%s"}' "$1" "$2"
}

partner_algo_for() {
    case "$1" in
        ModularAlgorithm)
            echo "PPO"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

share_latent_args() {
    case "$1" in
        ADAP|ADAP_MULT)
            echo "--share-latent"
            ;;
        *)
            echo ""
            ;;
    esac
}

layout_dir() {
    echo "${RESULTS_DIR}/$1/$2"
}

ego_model_base() {
    echo "$(layout_dir "$1" "$2")/models/ego"
}

alt_model_base() {
    echo "$(layout_dir "$1" "$2")/models/alt"
}

best_or_final() {
    base="$1"
    if [ -f "${base}-best.zip" ]; then
        echo "${base}-best"
    else
        echo "${base}-final"
    fi
}

train_one() {
    algo="$1"
    layout="$2"
    partner_algo="$(partner_algo_for "${algo}")"
    out_dir="$(layout_dir "${algo}" "${layout}")"
    models_dir="${out_dir}/models"
    logs_dir="${out_dir}/logs"
    mkdir -p "${models_dir}" "${logs_dir}"

    ego_base="${models_dir}/ego"
    alt_base="${models_dir}/alt"
    tensorboard_name="${algo}_${layout}_seed${SEED}"

    if [ "${FORCE}" != "1" ] && \
       { [ -f "${ego_base}-best.zip" ] || [ -f "${ego_base}-final.zip" ]; } && \
       { [ -f "${alt_base}-best.zip" ] || [ -f "${alt_base}-final.zip" ]; }; then
        echo "==> Skipping ${algo} on ${layout}: model already exists"
        return 0
    fi

    echo "==> Training ${algo} on ${layout} with partner ${partner_algo}"
    # shellcheck disable=SC2206
    extra_args=($(share_latent_args "${algo}"))

    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" trainer.py \
        OvercookedMultiEnv-v0 "${algo}" "${partner_algo}" \
        --env-config "$(json_config "${layout}")" \
        --seed "${SEED}" \
        --total-timesteps "${TIMESTEPS}" \
        --framestack "${FRAMES}" \
        --tensorboard-log "${logs_dir}" \
        --tensorboard-name "${tensorboard_name}" \
        --eval-freq "${EVAL_FREQ}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --ego-save "${ego_base}-final" \
        --alt-save "${alt_base}-final" \
        --best-ego-save "${ego_base}-best" \
        --best-alt-save "${alt_base}-best" \
        "${extra_args[@]}"
}

gif_one() {
    algo="$1"
    layout="$2"
    partner_algo="$(partner_algo_for "${algo}")"
    ego_base="$(best_or_final "$(ego_model_base "${algo}" "${layout}")")"
    alt_base="$(best_or_final "$(alt_model_base "${algo}" "${layout}")")"
    gifs_dir="${RESULTS_DIR}/gifs"
    mkdir -p "${gifs_dir}"

    if [ ! -f "${ego_base}.zip" ] || [ ! -f "${alt_base}.zip" ]; then
        echo "Skipping GIF for ${algo}/${layout}: missing model"
        return 0
    fi

    echo "==> Rendering GIF for ${algo} on ${layout}"
    "${PYTHON_BIN}" render_policy_gif.py OvercookedMultiEnv-v0 \
        --env-config "$(json_config "${layout}")" \
        --ego-config "$(model_config "${algo}" "${ego_base}")" \
        --partner-config "$(model_config "${partner_algo}" "${alt_base}")" \
        --framestack "${FRAMES}" \
        --output "${gifs_dir}/${algo}_${layout}.gif" \
        --fps "${GIF_FPS}" \
        --actions-output "${gifs_dir}/${algo}_${layout}.json"
}

train_all() {
    for algo in ${ALGORITHMS}; do
        for layout in ${LAYOUTS}; do
            train_one "${algo}" "${layout}"
        done
    done
}

gifs_all() {
    for algo in ${ALGORITHMS}; do
        for layout in ${LAYOUTS}; do
            gif_one "${algo}" "${layout}"
        done
    done
}

latest_all() {
    for algo in ${ALGORITHMS}; do
        echo "${algo}"
        for layout in ${LAYOUTS}; do
            ego_base="$(best_or_final "$(ego_model_base "${algo}" "${layout}")")"
            alt_base="$(best_or_final "$(alt_model_base "${algo}" "${layout}")")"
            if [ -f "${ego_base}.zip" ] || [ -f "${alt_base}.zip" ]; then
                echo "  ${layout}: ego=${ego_base}.zip alt=${alt_base}.zip"
            fi
        done
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
        echo "Usage: bash run.sh {train|gifs|all|latest}"
        echo
        echo "Examples:"
        echo "  bash run.sh train"
        echo "  bash run.sh gifs"
        echo "  TIMESTEPS=100000 SEED=11 bash run.sh all"
        echo "  FORCE=1 ALGORITHMS='PPO' LAYOUTS='simple' bash run.sh train"
        echo "  ALGORITHMS='PPO DQN A2C' LAYOUTS='simple random0' bash run.sh train"
        echo "  PYTHON_BIN=/data/luoey/conda_envs/overcooked/bin/python bash run.sh all"
        ;;
esac
