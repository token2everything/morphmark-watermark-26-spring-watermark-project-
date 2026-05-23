#!/bin/bash
# Sequential evaluation of all watermark methods
set -e
SAMPLES=${1:-50}
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

for config in configs/kgw.yaml configs/morphmark_exp.yaml configs/morphmark_linear.yaml configs/morphmark_log.yaml; do
    echo "=== $(date) Running $config ==="
    .venv/bin/python scripts/evaluate.py --config "$config" --n_samples "$SAMPLES"
    echo "=== $(date) Done $config ==="
done

echo "=== All done! ==="
