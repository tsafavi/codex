#!/bin/bash
KGE_DIR=$1

declare -a sizes=("s" "m" "l")
for size in "${sizes[@]}"; do
    codex="codex-${size}"
    dst="${KGE_DIR}/data/${codex}"
    mkdir ${dst}
    cp data/triples/${codex}/{train.txt,valid.txt,test.txt} ${dst}
done

cd ${KGE_DIR}/data/
for size in "${sizes[@]}"; do
    python preprocess.py codex-${size}
done
