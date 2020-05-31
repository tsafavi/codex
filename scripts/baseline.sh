#!/bin/bash
set -e

codex_dir="models/link-prediction/codex-m/complex"
fb_dir="models/link-prediction/fb15k-237/rescal"

python download_pretrained.py m link-prediction complex

if [ ! -f ${fb_dir}/checkpoint_best.pt ]; then
    curl http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-rescal.pt -o ${fb_dir}/checkpoint_best.pt
else
    echo "Skipping download of ${fb_dir}/checkpoint_best.pt because path already exists"
fi

python scripts/baseline.py ${codex_dir}/checkpoint_best.pt --csv codex.csv
python scripts/baseline.py ${fb_dir}/checkpoint_best.pt --csv fb.csv
