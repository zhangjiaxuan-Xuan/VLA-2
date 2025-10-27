#cd path/to/your/Agentic_VLA
set -x
set -e
zsh path/to/your/experiments/robot/libero_run/mps_start.sh
NLTK_DATA=path/to/your/GroundingDINO/mmdetection/nltk_data
SERVICE_PY="path/to/your/experiments/robot/libero_run/vision_planner_service.py"
MAIN_PY="path/to/your/experiments/robot/libero_run/main_agent_clean.py"
CONDA_ENV="ftdino"
ENDPOINT="ipc:///tmp/vision_planner.sock"
MODEL="path/to/your/openvla/pretrained_checkpoint_for_10"
TASK="libero_10"
WANDB_ENTITY="agent_eval"
# ---- New: timestamped log directory ----
BASE_LOG_DIR="path/to/your/experiments/cache/log/eval/10_time"
RUN_ID="$(date +'%Y%m%d_%H%M%S')"
RUN_DIR="${BASE_LOG_DIR}/${RUN_ID}"
mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "${BASE_LOG_DIR}/latest"   # point to the latest run
echo "[SH] Log directory: $RUN_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
SERVICE_LOG="${RUN_DIR}/service.log"
MAIN_LOG="${RUN_DIR}/main.log"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "[SH] Activating conda env: $CONDA_ENV"
source path/to/your/miniconda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

echo "[SH] Starting VisionPlannerService ..."
python -u "$SERVICE_PY" --endpoint "$ENDPOINT" --device cuda:0 >"$SERVICE_LOG" 2>&1 &
SERVICE_PID=$!

# Wait for the service to report READY
echo "[SH] Waiting for service to be ready ..."
until grep -q "\[Service\] READY" "$SERVICE_LOG"; do
    # If service process exited early, exit with error
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
        echo "[SH] Service process exited early. Check log: $SERVICE_LOG"
        exit 1
    fi
    sleep 2
done
echo "[SH] Service is READY."

conda activate path/to/your/conda_env/openvla

echo "[SH] Starting main process ..."
python -u "$MAIN_PY" --pretrained_checkpoint "$MODEL" --task_suite_name "$TASK" --num_trials_per_task=20 --thr=0.6 --wandb_entity "$WANDB_ENTITY" >>"$MAIN_LOG" 2>&1 &
MAIN_PID=$!

cleanup() {
    echo "[SH] Caught signal, cleaning up ..."
    kill -TERM $MAIN_PID 2>/dev/null || true
    kill -TERM $SERVICE_PID 2>/dev/null || true
    wait $MAIN_PID 2>/dev/null || true
    wait $SERVICE_PID 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

wait $MAIN_PID
