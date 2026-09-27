#!/usr/bin/env bash
# ==============================================================================
# CrystalFlow Server Execution Wrapper Script (run_server.sh)
# ==============================================================================
# Purpose:
#   Provides 1-command interfaces for Environment Check, Training, Sampling,
#   Evaluation, and Metric Calculation on GPU servers (e.g., Kyutech Cluster).
#
# Usage:
#   ./run_server.sh <command> [options]
#
# Commands:
#   check     - Validate GPU, CUDA, PyTorch, PyG and environment variables
#   train     - Launch CrystalFlow training (CSP or DNG) with Hydra & Lightning
#   sample    - Sample crystal structures from arbitrary chemical formulas
#   eval      - Run generation / CSP reconstruction evaluation on datasets
#   metrics   - Compute validity, novelty, match rate, and property metrics
#   help      - Show this help message
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ------------------------------------------------------------------------------
# Color Output Configuration
# ------------------------------------------------------------------------------
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO $(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN $(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR $(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" >&2
}

log_header() {
    echo -e "\n${BLUE}======================================================================${NC}"
    echo -e "${BLUE}  $*${NC}"
    echo -e "${BLUE}======================================================================${NC}\n"
}

# ------------------------------------------------------------------------------
# Environment Setup and Verification
# ------------------------------------------------------------------------------
ensure_env() {
    mkdir -p log hydra results/samples

    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.template" ]]; then
            log_warn ".env file not found. Creating default .env from .env.template..."
            cp .env.template .env
            # Set default PROJECT_ROOT to current directory
            sed -i "s|/home/<YOURHOME>/CrystalFlow|${SCRIPT_DIR}|g" .env 2>/dev/null || true
            log_info "Generated default .env with PROJECT_ROOT=${SCRIPT_DIR}"
        else
            log_warn "Neither .env nor .env.template found. Setting fallback environment variables."
            export PROJECT_ROOT="${SCRIPT_DIR}"
            export HYDRA_JOBS="${SCRIPT_DIR}/hydra"
            export WANDB_DIR="${SCRIPT_DIR}/log"
        fi
    fi

    if [[ -f ".env" ]]; then
        set -a
        source .env
        set +a
    fi
}

# ------------------------------------------------------------------------------
# Command: check (Environment & CUDA Validation)
# ------------------------------------------------------------------------------
cmd_check() {
    log_header "CrystalFlow Server Environment & Hardware Diagnostic"
    ensure_env

    echo "1. System & Python:"
    uname -a
    echo "Python Path: $(which python || echo 'Not Found')"
    python --version || true

    echo -e "\n2. GPU & Driver Status:"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    else
        log_warn "nvidia-smi not detected in PATH. (Running on CPU node?)"
    fi

    echo -e "\n3. PyTorch & CUDA Detection:"
    python -c "
import torch
print(f'PyTorch Version    : {torch.__version__}')
print(f'CUDA Available     : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version       : {torch.version.cuda}')
    print(f'Device Count       : {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  [GPU {i}] {torch.cuda.get_device_name(i)}')
" 2>&1 || log_error "Failed to import torch. Please install PyTorch first."

    echo -e "\n4. PyG (PyTorch Geometric) Core & C++ Extensions:"
    python -c "
import torch_geometric
print(f'PyG Version        : {torch_geometric.__version__}')
for ext in ['torch_scatter', 'torch_sparse', 'torch_cluster', 'torch_spline_conv', 'pyg_lib']:
    try:
        mod = __import__(ext)
        print(f'{ext:<18} : Available (v{getattr(mod, \"__version__\", \"N/A\")})')
    except ImportError:
        print(f'{ext:<18} : NOT INSTALLED')
" 2>&1 || log_warn "Some PyG extensions might be missing."

    echo -e "\n5. Crystallography & General ML Packages:"
    python -c "
for pkg in ['pymatgen', 'ase', 'pyxtal', 'smact', 'matminer', 'spglib', 'lightning', 'hydra', 'torchdyn']:
    try:
        mod = __import__(pkg)
        print(f'{pkg:<18} : Available (v{getattr(mod, \"__version__\", \"N/A\")})')
    except ImportError:
        print(f'{pkg:<18} : NOT INSTALLED')
" 2>&1 || true

    log_info "Diagnostic check complete."
}

# ------------------------------------------------------------------------------
# Command: train (Model Training)
# ------------------------------------------------------------------------------
cmd_train() {
    ensure_env

    TASK="csp"
    DATA="mp_20"
    GPUS="0"
    EPOCHS="3000"
    BATCH_SIZE=""
    LR="1e-3"
    EXPNAME=""
    BACKGROUND=false
    WANDB_MODE="offline"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --task) TASK="$2"; shift 2 ;;
            --data) DATA="$2"; shift 2 ;;
            --gpus) GPUS="$2"; shift 2 ;;
            --epochs) EPOCHS="$2"; shift 2 ;;
            --batch_size) BATCH_SIZE="$2"; shift 2 ;;
            --lr) LR="$2"; shift 2 ;;
            --expname) EXPNAME="$2"; shift 2 ;;
            --wandb) WANDB_MODE="$2"; shift 2 ;;
            --bg|--background) BACKGROUND=true; shift ;;
            *) log_error "Unknown train option: $1"; exit 1 ;;
        esac
    done

    # Count number of devices
    IFS=',' read -ra GPU_LIST <<< "${GPUS}"
    NUM_DEVICES=${#GPU_LIST[@]}

    TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
    if [[ -z "${EXPNAME}" ]]; then
        EXPNAME="${TASK^^}-${DATA}-${TIMESTAMP}"
    fi

    LOG_FILE="log/${EXPNAME}.log"

    # Select base model configuration by task
    case "${TASK}" in
        csp)
            MODEL_CFG="model=flow_polar"
            EXTRA_ARGS="+model.lattice_polar_sigma=0.1 model.cost_coord=10 model.cost_lattice=1"
            ;;
        dng)
            MODEL_CFG="model=flow_polar_w_type +model.type_encoding=table"
            EXTRA_ARGS="+model.lattice_polar_sigma=0.1 model.cost_type=10 model.cost_coord=10 model.cost_lattice=1"
            ;;
        dng-eform)
            DATA="mp_20_chgnet"
            MODEL_CFG="model=flow_polar_w_type +model.type_encoding=table +model.guide_threshold=-1"
            EXTRA_ARGS="+model.lattice_polar_sigma=0.1 model.cost_type=10 model.cost_coord=10 model.cost_lattice=1"
            ;;
        *)
            log_error "Unsupported task: '${TASK}'. Choose from 'csp', 'dng', or 'dng-eform'."
            exit 1
            ;;
    esac

    # Build DDP multi-device arguments if NUM_DEVICES > 1
    TRAINER_ARGS="train.pl_trainer.devices=${NUM_DEVICES}"
    if [[ ${NUM_DEVICES} -gt 1 ]]; then
        TRAINER_ARGS="${TRAINER_ARGS} +train.pl_trainer.strategy=ddp_find_unused_parameters_true"
    fi

    # Optional batch size override
    BS_ARG=""
    if [[ -n "${BATCH_SIZE}" ]]; then
        BS_ARG="data.datamodule.batch_size.train=${BATCH_SIZE}"
    fi

    log_header "Launching CrystalFlow Training: ${EXPNAME}"
    echo "Task        : ${TASK}"
    echo "Dataset     : ${DATA}"
    echo "GPUs        : ${GPUS} (${NUM_DEVICES} device(s))"
    echo "Max Epochs  : ${EPOCHS}"
    echo "Learning Rate: ${LR}"
    echo "WandB Mode  : ${WANDB_MODE}"
    echo "Log Output  : ${LOG_FILE}"

    CMD="CUDA_VISIBLE_DEVICES=${GPUS} HYDRA_FULL_ERROR=1 python diffcsp/run.py \
data=${DATA} data.train_max_epochs=${EPOCHS} \
${MODEL_CFG} \
${TRAINER_ARGS} \
optim.optimizer.lr=${LR} \
optim.optimizer.weight_decay=0 \
optim.lr_scheduler.factor=0.6 \
model.decoder.num_freqs=256 \
model.decoder.rec_emb=sin model.decoder.num_millers=8 \
+model.decoder.na_emb=0 \
model.decoder.hidden_dim=512 model.decoder.num_layers=6 \
logging.wandb.mode=${WANDB_MODE} \
logging.wandb.project=crystalflow \
expname=${EXPNAME} \
${EXTRA_ARGS} \
${BS_ARG}"

    if [[ "${BACKGROUND}" == true ]]; then
        log_info "Running in background (nohup)..."
        nohup bash -c "${CMD}" > "${LOG_FILE}" 2>&1 &
        PID=$!
        log_info "Process started with PID ${PID}."
        log_info "Monitor progress with: tail -f ${LOG_FILE}"
    else
        log_info "Running in foreground..."
        eval "${CMD}" 2>&1 | tee "${LOG_FILE}"
    fi
}

# ------------------------------------------------------------------------------
# Command: sample (Generation from Arbitrary Formula)
# ------------------------------------------------------------------------------
cmd_sample() {
    ensure_env

    MODEL_PATH=""
    FORMULA=""
    NUM_EVALS=10
    ODE_STEPS=100
    SAVE_PATH=""
    BATCH_SIZE=500
    GPUS="0"
    SAVE_TRAJ=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--model_path) MODEL_PATH="$2"; shift 2 ;;
            -f|--formula) FORMULA="$2"; shift 2 ;;
            -n|--num_evals) NUM_EVALS="$2"; shift 2 ;;
            -N|--ode_steps) ODE_STEPS="$2"; shift 2 ;;
            -d|--save_path) SAVE_PATH="$2"; shift 2 ;;
            -B|--batch_size) BATCH_SIZE="$2"; shift 2 ;;
            --gpus) GPUS="$2"; shift 2 ;;
            --traj) SAVE_TRAJ=true; shift ;;
            *) log_error "Unknown sample option: $1"; exit 1 ;;
        esac
    done

    if [[ -z "${MODEL_PATH}" ]]; then
        log_error "Missing required option: --model_path (-m)"
        exit 1
    fi
    if [[ -z "${FORMULA}" ]]; then
        log_error "Missing required option: --formula (-f) (e.g., -f YMnO3)"
        exit 1
    fi

    if [[ -z "${SAVE_PATH}" ]]; then
        SAVE_PATH="results/samples/${FORMULA}"
    fi
    mkdir -p "${SAVE_PATH}"

    TRAJ_ARG=""
    if [[ "${SAVE_TRAJ}" == true ]]; then
        TRAJ_ARG="--traj"
    fi

    log_header "Sampling Crystal Structures: ${FORMULA}"
    echo "Model Path : ${MODEL_PATH}"
    echo "Formula    : ${FORMULA}"
    echo "Samples (N): ${NUM_EVALS}"
    echo "ODE Steps  : ${ODE_STEPS}"
    echo "Save Path  : ${SAVE_PATH}"
    echo "GPU        : ${GPUS}"

    CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/sample.py \
        --model_path "${MODEL_PATH}" \
        --save_path "${SAVE_PATH}" \
        --formula "${FORMULA}" \
        --num_evals "${NUM_EVALS}" \
        --ode-int-steps "${ODE_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --anneal_coords --anneal_slope 5 \
        ${TRAJ_ARG}

    log_info "Sampling completed. Output files stored in: ${SAVE_PATH}"
}

# ------------------------------------------------------------------------------
# Command: eval (Dataset Evaluation)
# ------------------------------------------------------------------------------
cmd_eval() {
    ensure_env

    MODEL_PATH=""
    DATASET="mp_20"
    TASK="csp"
    LABEL="eval_$(date +'%Y%m%d_%H%M%S')"
    ODE_STEPS=100
    NUM_EVALS=1
    GPUS="0"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--model_path) MODEL_PATH="$2"; shift 2 ;;
            --dataset) DATASET="$2"; shift 2 ;;
            --task) TASK="$2"; shift 2 ;;
            --label) LABEL="$2"; shift 2 ;;
            -N|--ode_steps) ODE_STEPS="$2"; shift 2 ;;
            -n|--num_evals) NUM_EVALS="$2"; shift 2 ;;
            --gpus) GPUS="$2"; shift 2 ;;
            *) log_error "Unknown eval option: $1"; exit 1 ;;
        esac
    done

    if [[ -z "${MODEL_PATH}" ]]; then
        log_error "Missing required option: --model_path (-m)"
        exit 1
    fi

    log_header "Running Evaluation: ${TASK^^} on ${DATASET}"
    echo "Model Path : ${MODEL_PATH}"
    echo "Task       : ${TASK}"
    echo "Dataset    : ${DATASET}"
    echo "Label      : ${LABEL}"
    echo "ODE Steps  : ${ODE_STEPS}"
    echo "Num Evals  : ${NUM_EVALS}"

    if [[ "${TASK}" == "csp" ]]; then
        MULTI_ARG=""
        if [[ ${NUM_EVALS} -gt 1 ]]; then
            MULTI_ARG="--num_evals ${NUM_EVALS}"
        fi
        CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/evaluate.py \
            --model_path "${MODEL_PATH}" \
            --ode-int-steps "${ODE_STEPS}" \
            --dataset "${DATASET}" \
            --anneal_coords --anneal_slope 5 \
            --label "${LABEL}" \
            ${MULTI_ARG}
    elif [[ "${TASK}" == "gen" || "${TASK}" == "dng" ]]; then
        CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/generation.py \
            --model_path "${MODEL_PATH}" \
            --ode-int-steps "${ODE_STEPS}" \
            --dataset "${DATASET}" \
            --label "${LABEL}"
    else
        log_error "Unknown task: ${TASK}. Choose 'csp' or 'gen'."
        exit 1
    fi

    log_info "Evaluation finished with label '${LABEL}'."
}

# ------------------------------------------------------------------------------
# Command: metrics (Calculate Metrics)
# ------------------------------------------------------------------------------
cmd_metrics() {
    ensure_env

    ROOT_PATH=""
    TASK="csp"
    GT_FILE=""
    LABEL=""
    MULTI_EVAL=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--model_path|--root_path) ROOT_PATH="$2"; shift 2 ;;
            --task|--tasks) TASK="$2"; shift 2 ;;
            --gt_file) GT_FILE="$2"; shift 2 ;;
            --label) LABEL="$2"; shift 2 ;;
            --multi_eval) MULTI_EVAL=true; shift ;;
            *) log_error "Unknown metrics option: $1"; exit 1 ;;
        esac
    done

    if [[ -z "${ROOT_PATH}" ]]; then
        log_error "Missing required option: --root_path (-m)"
        exit 1
    fi
    if [[ -z "${LABEL}" ]]; then
        log_error "Missing required option: --label"
        exit 1
    fi
    if [[ -z "${GT_FILE}" ]]; then
        GT_FILE="data/mp_20/test.csv"
        log_warn "No --gt_file specified, defaulting to ${GT_FILE}"
    fi

    log_header "Computing Metrics"
    echo "Root Path : ${ROOT_PATH}"
    echo "Task      : ${TASK}"
    echo "GT File   : ${GT_FILE}"
    echo "Label     : ${LABEL}"

    MULTI_FLAG=""
    if [[ "${MULTI_EVAL}" == true ]]; then
        MULTI_FLAG="--multi_eval"
    fi

    python scripts/compute_metrics.py \
        --root_path "${ROOT_PATH}" \
        --tasks "${TASK}" \
        --gt_file "${GT_FILE}" \
        --label "${LABEL}" \
        ${MULTI_FLAG}

    log_info "Metric computation complete."
}

# ------------------------------------------------------------------------------
# Command: help
# ------------------------------------------------------------------------------
cmd_help() {
    cat << 'EOF'
CrystalFlow Server Execution Wrapper (run_server.sh)

Usage:
  ./run_server.sh <command> [options]

Available Commands:
  check
      Run environment, GPU (nvidia-smi), CUDA, PyTorch, PyG, and dependencies check.

  train
      Launch training on GPU server.
      Options:
        --task <csp|dng|dng-eform>  Task type (default: csp)
        --data <dataset>            Dataset name (default: mp_20)
        --gpus <0|0,1,2,3>          GPU devices to allocate (default: 0)
        --epochs <int>              Max training epochs (default: 3000)
        --batch_size <int>          Override train batch size (optional)
        --lr <float>                Initial learning rate (default: 1e-3)
        --expname <name>            Experiment name (default: auto-generated)
        --wandb <mode>              WandB mode: online/offline/disabled (default: offline)
        --bg, --background          Run in background with nohup

  sample
      Sample crystal structures from arbitrary chemical composition.
      Options:
        -m, --model_path <path>     Directory of trained model/checkpoint (REQUIRED)
        -f, --formula <formula>     Chemical formula, e.g. YMnO3, BaTiO3 (REQUIRED)
        -n, --num_evals <int>       Number of structures to generate (default: 10)
        -N, --ode_steps <int>       ODE integration steps (default: 100)
        -d, --save_path <dir>       Output directory (default: results/samples/<formula>)
        -B, --batch_size <int>      Batch size for sampling (default: 500)
        --gpus <0>                  GPU device index (default: 0)
        --traj                      Also save trajectory (XDATCAR)

  eval
      Evaluate model generation or CSP reconstruction on datasets.
      Options:
        -m, --model_path <path>     Directory of trained model (REQUIRED)
        --task <csp|gen>            Evaluation task (default: csp)
        --dataset <name>            Dataset name (default: mp_20)
        --label <string>            Evaluation run label (default: timestamped)
        -N, --ode_steps <int>       ODE steps (default: 100)
        -n, --num_evals <int>       Number of evaluations per sample (default: 1)
        --gpus <0>                  GPU device index (default: 0)

  metrics
      Compute evaluation metrics (Match rate, RMSD, S.U.N., validity).
      Options:
        -m, --root_path <path>      Model output directory (REQUIRED)
        --label <string>            Evaluation label used in eval step (REQUIRED)
        --task <csp|gen>            Task type (default: csp)
        --gt_file <path>            Ground truth CSV (default: data/mp_20/test.csv)
        --multi_eval                Enable multi-evaluation mode

  help
      Show this help documentation.

Examples:
  # Check environment & CUDA
  ./run_server.sh check

  # Train CSP model on GPU 0 in background
  ./run_server.sh train --task csp --data mp_20 --gpus 0 --bg

  # Sample 20 candidates for YMnO3
  ./run_server.sh sample -m hydra/singlerun/CSP-mp20 -f YMnO3 -n 20
EOF
}

# ------------------------------------------------------------------------------
# Main Dispatcher
# ------------------------------------------------------------------------------
COMMAND="${1:-help}"
shift || true

case "${COMMAND}" in
    check)   cmd_check "$@" ;;
    train)   cmd_train "$@" ;;
    sample)  cmd_sample "$@" ;;
    eval)    cmd_eval "$@" ;;
    metrics) cmd_metrics "$@" ;;
    help|-h|--help) cmd_help ;;
    *)
        log_error "Unknown command: '${COMMAND}'"
        echo "Run './run_server.sh help' for usage instructions."
        exit 1
        ;;
esac
