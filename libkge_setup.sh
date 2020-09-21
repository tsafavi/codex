#!/bin/bash
set -e

# -----------------------------------------
# LibKGE setup from https://github.com/uma-pi1/kge/blob/master/data/download_all.sh
git clone https://github.com/uma-pi1/kge.git
cd kge
git checkout a9ecd249ec2d205df59287f64553a1536add4a43  # freeze but leave the git history
pip install -e .

cd data/

if [ ! -d "fb15k-237" ]; then
	echo Downloading fb15k-237
	curl -O https://www.uni-mannheim.de/media/Einrichtungen/dws/pi1/kge_datasets/fb15k-237.tar.gz
	tar xvf fb15k-237.tar.gz
else
	echo fb15k-237 already present
fi
if [ ! -f "fb15k-237/dataset.yaml" ]; then
 	python preprocess.py fb15k-237
else
	echo fb15k-237 already prepared
fi
# -----------------------------------------

declare -a sizes=("s" "m" "l")
for size in "${sizes[@]}"; do
    codex="codex-${size}"
    mkdir ${codex}
    cp ../../data/triples/${codex}/{train,valid,test}.txt ${codex}
done

for size in "${sizes[@]}"; do
    python preprocess.py codex-${size}
done
