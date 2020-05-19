#!/bin/bash
models=$(find models/triple-classification/codex-s/*/*.pt)
python scripts/tc.py ${models} --negative-types true_neg --csv tc.csv
