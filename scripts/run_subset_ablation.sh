#!/usr/bin/env bash
set -euo pipefail

SIZE=${1:-tiny}
N_PATIENTS=${2:-10000}
FORCE=false
INFERENCE_PATIENTS=""
N_SAMPLES=""
CONFIG=config/subset_run.yaml
ROUTES=("no_ecg" "fusion_class" "all_branches")

# Parse flags from any argument position
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --inference-patients=*) INFERENCE_PATIENTS="${arg#*=}" ;;
        --n-samples=*) N_SAMPLES="${arg#*=}" ;;
    esac
done

# output_dir is always ./data/processed, but --n-patients appends /n{N}
OUT="./data/processed/n${N_PATIENTS}"

SEQ="$OUT/sequences.parquet"
TOK="$OUT/tokenized/all_tokens.parquet"

echo "=== Subset ablation: size=$SIZE, n_patients=$N_PATIENTS ==="

# Step 1-2: Extract (once — route-independent)
if [[ "$FORCE" == true ]] || [[ ! -f "$SEQ" ]]; then
    echo "==> Extracting..."
    protoecg-pipeline extract --config "$CONFIG" --n-patients "$N_PATIENTS" --workers 0
else
    echo "==> Skipping extract (found $SEQ). Use --force to re-run."
fi

# Step 3: Tokenize (once — route-independent)
if [[ "$FORCE" == true ]] || [[ ! -f "$TOK" ]]; then
    echo "==> Tokenizing..."
    protoecg-pipeline tokenize --config "$CONFIG" --n-patients "$N_PATIENTS"
else
    echo "==> Skipping tokenize (found $TOK). Use --force to re-run."
fi

# Step 4-5: Train + Evaluate per route
for ROUTE in "${ROUTES[@]}"; do
    CKPT="$OUT/checkpoints/${ROUTE}_${SIZE}/final"
    METRICS="$OUT/evaluation/${ROUTE}_${SIZE}/metrics.json"

    if [[ "$FORCE" == true ]] || [[ ! -d "$CKPT" ]]; then
        echo "==> Training route=$ROUTE size=$SIZE..."
        protoecg-pipeline train --config "$CONFIG" --size "$SIZE" --route "$ROUTE" --n-patients "$N_PATIENTS"
    else
        echo "==> Skipping train $ROUTE (found $CKPT). Use --force to re-run."
    fi

    EVAL_ARGS=(--config "$CONFIG" --size "$SIZE" --route "$ROUTE" --n-patients "$N_PATIENTS")
    [[ -n "$INFERENCE_PATIENTS" ]] && EVAL_ARGS+=(--inference-patients "$INFERENCE_PATIENTS")
    [[ -n "$N_SAMPLES" ]] && EVAL_ARGS+=(--n-samples "$N_SAMPLES")

    if [[ "$FORCE" == true ]] || [[ ! -f "$METRICS" ]]; then
        echo "==> Evaluating route=$ROUTE size=$SIZE..."
        protoecg-pipeline evaluate "${EVAL_ARGS[@]}"
    else
        echo "==> Skipping evaluate $ROUTE (found $METRICS). Use --force to re-run."
    fi
done
