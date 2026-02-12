#!/usr/bin/env bash
set -euo pipefail

CONFIG=config/full_run.yaml
ROUTE=${1:-all_branches}
SIZE=${2:-small}

echo "=== CLIF ProtoECG FM Full Pipeline ==="
echo "Config: $CONFIG | Route: $ROUTE | Size: $SIZE"
echo ""

# Step 1-2: Extract CLIF data → MEDS sequences + labels + ECG tokens
echo ">>> Step 1-2: Extract"
protoecg-pipeline extract --config "$CONFIG" --workers 0

# Step 3: Tokenize (build vocab + convert to token IDs)
echo ">>> Step 3: Tokenize"
protoecg-pipeline tokenize --config "$CONFIG"

# Step 4: Train
echo ">>> Step 4: Train ($ROUTE / $SIZE)"
protoecg-pipeline train --config "$CONFIG" --size "$SIZE" --route "$ROUTE"

# Step 5: Evaluate
echo ">>> Step 5: Evaluate"
protoecg-pipeline evaluate --config "$CONFIG" --size "$SIZE" --route "$ROUTE"

echo ""
echo "=== Pipeline complete ==="
