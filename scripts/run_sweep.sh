#!/bin/bash
set -e
for cfg in configs/*_*hz.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "=== $name ==="
  python rl/train.py --config "$cfg"
  python scripts/measure_latency.py \
    --model "logs/${name}/final_model.zip" \
    --config "$cfg" \
    --n-steps 2000
done
