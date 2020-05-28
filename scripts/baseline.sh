#!/bin/bash
codex_dir="models/link-prediction/codex-m/complex"
fb_dir="models/link-prediction/fb15k-237"
curl http://web.informatik.uni-mannheim.de/pi1/iclr2020-models/fb15k-237-rescal.pt -o ${fb_dir}/checkpoint_best.pt
python scripts/baseline.py ${codex_dir}/checkpoint_best.pt --csv codex.csv
python scripts/baseline.py ${fb_dir}/checkpoint_best.pt --csv fb.csv
