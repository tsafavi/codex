#!/bin/bash
set -e

# retrieve and install libkge in development mode
git clone https://github.com/uma-pi1/kge.git
cd kge
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

declare -a sizes=("s" "m" "l")
for size in "${sizes[@]}"; do
    codex="codex-${size}"
    mkdir ${codex}
    cp ../../data/triples/${codex}/{train.txt,valid.txt,test.txt} ${codex}
done

for size in "${sizes[@]}"; do
    python preprocess.py codex-${size}
done
