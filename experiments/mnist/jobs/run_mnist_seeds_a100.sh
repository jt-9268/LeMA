#!/bin/bash
#SBATCH --job-name=mnist-lema-seeds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --chdir=/scratch/jtaylor/mnist
#SBATCH --output=/scratch/jtaylor/%x_%j.out
#SBATCH --error=/scratch/jtaylor/%x_%j.err
#
# train.py with LeMA, B = 8192 (4 GPUs × block 2048), EPOCHS (default 5), seeds 0-4, on 4×A100.
# Variant via environment variables (default ELU + He, LeMA damping defaults, no smoothing):
#   ACTIVATION, INIT, DAMP_START, DAMP_MIN, LABEL_SMOOTHING, EPOCHS, SEEDS, SHARD, SLICE
# Calibration (temperature, ECE, Brier) is logged at every evaluation by train.py.
# --seed sets the initial weights and the data order: same seed ⇒ same batches in every variant.
#
#   sbatch --export=ALL,ACTIVATION=relu,INIT=default                       run_mnist_seeds_a100.sh
#   sbatch --export=ALL,ACTIVATION=relu,INIT=default,LABEL_SMOOTHING=0.01  run_mnist_seeds_a100.sh
#   sbatch --export=ALL,ACTIVATION=relu,INIT=default,LABEL_SMOOTHING=0.03  run_mnist_seeds_a100.sh
#
# Resubmit-safe: a seed whose log has the final "epoch info" line (and a matching header) is
# skipped; an incomplete seed is rerun from scratch.

set -eo pipefail

source /scratch/jtaylor/miniforge3/bin/activate /scratch/jtaylor/envs/torch312
set -u

CODE=/scratch/jtaylor/mnist                       # train.py, summarize_seeds.py, compare_runs.py
RUNS=/scratch/jtaylor/runs
DATA_DIR=${DATA_DIR:-$CODE/.data}                 # must contain MNIST/raw/*-ubyte
export PYTHONPATH=/scratch/jtaylor/LeMA/experiments:/scratch/jtaylor/LeMA${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=8 NCCL_DEBUG=WARN PYTHONUNBUFFERED=1   # 32 CPUs / 4 GPU processes

ACTIVATION=${ACTIVATION:-elu}
INIT=${INIT:-he}
tag() { LC_ALL=C printf '%g' "$1"; }             # same formatting as train.py's "{:g}"
DAMP_START=${DAMP_START:-}; DS_TAG=""; DS_ARGS=()
[[ -n "$DAMP_START" ]] && { DS_TAG=$(tag "$DAMP_START"); DS_ARGS=(--damp-start "$DAMP_START"); }
DAMP_MIN=${DAMP_MIN:-}; DM_TAG=""; DM_ARGS=()
[[ -n "$DAMP_MIN" ]] && { DM_TAG=$(tag "$DAMP_MIN"); DM_ARGS=(--damp-min "$DAMP_MIN"); }
LABEL_SMOOTHING=${LABEL_SMOOTHING:-}; LS_TAG=""; LS_ARGS=()
[[ -n "$LABEL_SMOOTHING" ]] && { LS_TAG=$(tag "$LABEL_SMOOTHING"); LS_ARGS=(--label-smoothing "$LABEL_SMOOTHING"); }
SEEDS=${SEEDS:-"0 1 2 3 4"}
EPOCHS=${EPOCHS:-5}
BLOCK=2048                                        # B = 4 × 2048 = 8192 (same as the H200 runs)

# Shard/slice sized to GPU memory: peak ≈ 5.05 × shard × 1.2M params × 4 B (49.6 GB at 2048).
gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
if (( gpu_mib >= 60000 )); then DEF_SHARD=2048; DEF_SLICE=1024; else DEF_SHARD=1024; DEF_SLICE=512; fi
SHARD=${SHARD:-$DEF_SHARD}
SLICE=${SLICE:-$DEF_SLICE}

RUN_DIR=${RUN_DIR:-$RUNS/mnist_${ACTIVATION}_${INIT}${DS_TAG:+_ds$DS_TAG}${DM_TAG:+_dm$DM_TAG}${LS_TAG:+_ls$LS_TAG}_b8192_${EPOCHS}ep_a100_seeds_v1}

for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
    [[ -s "$DATA_DIR/MNIST/raw/$f" ]] || { echo "ERROR: missing $DATA_DIR/MNIST/raw/$f" >&2; exit 2; }
done
for f in train.py summarize_seeds.py; do
    [[ -f "$CODE/$f" ]] || { echo "ERROR: missing $CODE/$f" >&2; exit 2; }
done

# Fail fast if train.py does not import or predates the options this job needs
help_out=$(python "$CODE/train.py" --help 2>&1) \
    || { echo "ERROR: train.py --help failed:" >&2; echo "$help_out" >&2; exit 2; }
for opt in --seed --activation --init --calibration ${DAMP_START:+--damp-start} ${DAMP_MIN:+--damp-min} \
           ${LABEL_SMOOTHING:+--label-smoothing}; do
    grep -q -- "$opt" <<< "$help_out" \
        || { echo "ERROR: $CODE/train.py has no $opt option; copy the current version" >&2; exit 2; }
done

mkdir -p "$RUN_DIR/logs"
{   echo "=== job ${SLURM_JOB_ID:-local} on $(hostname -s) at $(date -Is)"
    echo "code md5: $(cd "$CODE" && md5sum train.py summarize_seeds.py | tr '\n' ' ')"
    echo "args: --activation $ACTIVATION --init $INIT ${DS_ARGS[*]:-} ${DM_ARGS[*]:-} ${LS_ARGS[*]:-} --block-size $BLOCK --shard-size $SHARD --slice-size $SLICE --epochs $EPOCHS seeds=[$SEEDS]"
    python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'tv', torchvision.__version__, '| GPUs', torch.cuda.device_count())"
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
} | tee -a "$RUN_DIR/meta.txt"
(cd "$CODE" && git diff HEAD > "$RUN_DIR/git_diff_${SLURM_JOB_ID:-local}.patch" 2>/dev/null) || true

for s in $SEEDS; do
    log="$RUN_DIR/logs/seed$s.log"
    if [[ -f "$log" ]] && grep -q "epoch info | epoch: $EPOCHS |" "$log"; then
        # header must match this variant; fields absent from older logs count as the defaults
        if [[ -n "$DS_TAG" ]]; then ds_re=" \| damp start: $DS_TAG"; else ds_re="( \| damp start: 0\.001)?"; fi
        if [[ -n "$DM_TAG" ]]; then dm_re=" \| damp min: $DM_TAG"; else dm_re="( \| damp min: 0\.1)?"; fi
        if [[ -n "$LS_TAG" ]]; then ls_re=" \| label smoothing: $LS_TAG"; else ls_re="( \| label smoothing: 0)?"; fi
        grep -qE "Activation: $ACTIVATION \| init: $INIT \| seed: $s$ds_re$dm_re$ls_re\$" "$log" \
            || { echo "ERROR: $log is complete but was not run with $ACTIVATION/$INIT/seed $s/damp start ${DS_TAG:-0.001}/damp min ${DM_TAG:-0.1}/label smoothing ${LS_TAG:-0}" >&2; exit 2; }
        echo "seed $s: complete, skipping"; continue
    fi
    echo "seed $s: start $(date -Is)"
    if ! torchrun --standalone --nproc_per_node=4 "$CODE/train.py" \
            --data-dir "$DATA_DIR" \
            --activation "$ACTIVATION" --init "$INIT" "${DS_ARGS[@]}" "${DM_ARGS[@]}" "${LS_ARGS[@]}" \
            --block-size $BLOCK --shard-size "$SHARD" --slice-size "$SLICE" \
            --epochs "$EPOCHS" --seed "$s" 2>&1 | tee "$log"; then
        echo "seed $s: FAILED (see $log)" | tee -a "$RUN_DIR/failures.txt"
    fi
done

python "$CODE/summarize_seeds.py" "$RUN_DIR"
echo "end=$(date -Is)"
