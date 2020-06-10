#!/bin/bash
set -e

declare -a sizes=("s" "m")
declare -a models=("rescal" "transe" "complex" "conve")

for size in "${sizes[@]}"; do
    # Download pretrained model if not already existing
    python download_pretrained.py ${size} triple-classification ${models[@]}

    for model in "${models[@]}"; do
        model_file="models/triple-classification/codex-${size}/${model}/checkpoint_best.pt"
        calib_type="isotonic"

        if [ ${model} = "transe" ] && [ ${size} = "m" ]; then
            calib_type="sigmoid"
        fi
        
        python scripts/tc.py ${model_file} --size ${size} --calib-type ${calib_type}
    done
done
