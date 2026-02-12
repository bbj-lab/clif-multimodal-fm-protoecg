#!/usr/bin/env bash
set -euo pipefail

CONFIG=config/full_run.yaml
SIZE=${1:-small}
FORCE=false
ROUTES=("no_ecg" "fusion_class" "all_branches")

# Parse --force flag from any argument position
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
    esac
done

# Read output_dir from config (default: ./data/processed)
OUT=$(python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get('data', {}).get('output_dir', './data/processed'))
")

SEQ="$OUT/sequences.parquet"
TOK="$OUT/tokenized/all_tokens.parquet"

# Step 1-2: Extract (once — route-independent)
if [[ "$FORCE" == true ]] || [[ ! -f "$SEQ" ]]; then
    echo "==> Extracting..."
    protoecg-pipeline extract --config "$CONFIG" --workers 0
else
    echo "==> Skipping extract (found $SEQ). Use --force to re-run."
fi

# Step 3: Tokenize (once — route-independent)
if [[ "$FORCE" == true ]] || [[ ! -f "$TOK" ]]; then
    echo "==> Tokenizing..."
    protoecg-pipeline tokenize --config "$CONFIG"
else
    echo "==> Skipping tokenize (found $TOK). Use --force to re-run."
fi

# Step 4-5: Train + Evaluate per route
for ROUTE in "${ROUTES[@]}"; do
    CKPT="$OUT/checkpoints/${ROUTE}_${SIZE}/final"
    METRICS="$OUT/evaluation/${ROUTE}_${SIZE}/metrics.json"

    if [[ "$FORCE" == true ]] || [[ ! -d "$CKPT" ]]; then
        echo "==> Training route=$ROUTE size=$SIZE..."
        protoecg-pipeline train --config "$CONFIG" --size "$SIZE" --route "$ROUTE"
    else
        echo "==> Skipping train $ROUTE (found $CKPT). Use --force to re-run."
    fi

    if [[ "$FORCE" == true ]] || [[ ! -f "$METRICS" ]]; then
        echo "==> Evaluating route=$ROUTE size=$SIZE..."
        protoecg-pipeline evaluate --config "$CONFIG" --size "$SIZE" --route "$ROUTE"
    else
        echo "==> Skipping evaluate $ROUTE (found $METRICS). Use --force to re-run."
    fi
done
