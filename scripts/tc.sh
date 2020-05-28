#!/bin/bash
declare -a sizes=("s" "m")

for size in "${sizes[@]}"; do
    models=$(find models/triple-classification/codex-${size}/*/*.pt)
    python scripts/tc.py ${models}
done
