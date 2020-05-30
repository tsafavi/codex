#!/bin/bash
set -e

declare -a sizes=("s" "m")

for size in "${sizes[@]}"; do
    python download_pretrained.py ${size} triple-classification rescal transe complex conve
    models=$(find models/triple-classification/codex-${size}/*/*.pt)
    python scripts/tc.py ${models} --size ${size}
done
