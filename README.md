<img src="codex_logo.png" width="450" />

CoDEx is a set of knowledge graph **Co**mpletion **D**atasets **Ex**tracted from Wikidata and Wikipedia.
As introduced and described by our EMNLP 2020 paper <a href="https://arxiv.org/pdf/2009.07810.pdf" target="_blank">CoDEx: A Comprehensive Knowledge Graph Completion Benchmark</a>,
CoDEx offers three rich knowledge graph datasets that contain positive and hard negative triples, entity types, entity and relation descriptions, and Wikipedia page extracts for entities. 
We provide baseline performance results, configuration files, and pretrained models
on CoDEx using the [LibKGE](https://github.com/uma-pi1/kge) framework for two knowledge graph completion tasks, link prediction and triple classification.

The statistics for each CoDEx dataset are as follows:

|          | Entities | Relations | Train   | Valid (+) | Test (+) | Valid (-) | Test (-) | Total triples |
|----------|---------:|----------:|--------:|----------:|---------:|----------:|---------:|--------------:|
| CoDEx-S  | 2,034    | 42        | 32,888  | 1,827     | 1,828    | 1,827     | 1,828    | 36,543        |
| CoDEx-M  | 17,050   | 51        | 185,584 | 10,310    | 10,311   | 10,310    | 10,311   | 206,205       |
| CoDEx-L  | 77,951   | 69        | 551,193 | 30,622    | 30,622   | -         | -        | 612,437       |
| Raw dump | 380,038  | 75        | -       | -         | -        | -         | -        | 1,156,222     |

**Note**: If you are interested in contributing to the CoDEx corpus, feel free to open an issue or a PR! 

## Table of contents
1. <a href="#quick-start">Quick start</a>
2. <a href="#explore">Data exploration and analysis</a>
3. <a href="#models">Pretrained models and results</a>
    - <a href="#kge">LibKGE setup</a>
    - <a href="#scripts">Reproducing our results</a>
      - <a href="#lp-script">Link prediction</a>
      - <a href="#tc-script">Triple classification</a>
      - <a href="#baseline-script">Comparison to FB15k-237</a>
    - <a href="#pretrained">Downloading pretrained models via the command line</a>
    - <a href="#lp">Link prediction results</a>
      - <a href="#s-lp">CoDEx-S</a>
      - <a href="#m-lp">CoDEx-M</a>
      - <a href="#l-lp">CoDEx-L</a>
    - <a href="#tc">Triple classification results</a>
      - <a href="#s-tc">CoDEx-S</a>
      - <a href="#m-tc">CoDEx-M</a>
4. <a href="#data">Data directory structure</a>
    - <a href="#entities">Entities and entity types</a>
    - <a href="#relations">Relations</a>
    - <a href="#triples">Triples</a>
5. <a href="#cite">How to cite</a>
6. <a href="#ref">References and acknowledgements</a>

## <a id="quick-start">Quick start</a>
If you'd like to download the CoDEx data, code, and/or pretrained models __locally to your machine__, run the following commands. 
If you only want to play with the data in a __remote environment__, head to the <a href="#explore">next section on data exploration and analysis</a>, and follow the instructions to view the CoDEx data with Colab. 
```
# unzip the repository
git clone https://github.com/tsafavi/codex.git
cd codex

# extract English Wikipedia plain-text excerpts for entities
# other language codes available: ar, de, es, ru, zh
./extract.sh en

# set up a virtual environment and install the Python requirements
python3.7 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# finally, install the codex data-loading API
pip install -e .
```

## <a id="explore">Data exploration and analysis</a>

To get familiar with the CoDEx datasets and the data-loading API in an easy-to-use interface, we have provided an exploration Jupyter notebook called `Explore CoDEx.ipynb`.  

You have two options for running the notebook: 

- __Run on Google Colab__: Open the <a href="https://colab.research.google.com/github/tsafavi/codex/blob/master/Explore%20CoDEx.ipynb" target="_blank">notebook on Google's Colab platform</a> and follow the instructions in the first cell to install all the requirements and data remotely. __Make sure to restart the Colab runtime after installing the requirements__ before you run any of the following cells. 
- __Run locally__: Run the following commands to register your virtual environment with JupyterLab and launch JupyterLab: 
  ```
  # run from codex/
  python -m ipykernel install --user --name=myenv
  jupyter lab
  ```
  Now, navigate to JupyterLab in your browser and open the ```Explore CoDEx.ipynb``` notebook in your browser.  

## <a id="models">Pretrained models and results</a>

### <a id="kge">LibKGE setup</a>

To **use the pretrained models or run any scripts that involve pretrained models**, you will need to set up [LibKGE](https://github.com/uma-pi1/kge).
Run the following: 
```
# run from codex/
# this may take a few minutes
./libkge_setup.sh
```
This script will install the library inside ```codex/kge/```, download the FB15K-237 dataset (which we use in our experiments) to ```kge/data/```, and copy each CoDEx dataset to ```kge/data/``` and preprocess each dataset according to the format the LibKGE requires. 

### <a id="scripts">Reproducing our results</a>

We provide evaluation scripts to reproduce results in our paper. You must have set up LibKGE using the <a href="#kge">instructions we provided</a>. 

#### <a id="lp-script">Link prediction</a>

```scripts/lp_gpu.sh``` and ```scripts/lp_cpu.sh``` run link prediction on all models and datasets using the LibKGE evaluation API.
To run on GPU:
```
# run from codex/
# this may take a few minutes
scripts/lp_gpu.sh  # change to lp_cpu.sh to run on CPU
```
Note that this script first downloads all link prediction models on CoDEx-S through L and saves them to ```models/link-prediction/codex-{s,m,l}/``` if they do not already exist.

#### <a id="tc-script">Triple classification</a>

```scripts/tc.sh``` runs triple classification and outputs validation and test accuracy/F1. 
To run:
```
# run from codex/
# this may take a few minutes
scripts/tc.sh  # runs on CPU
```
Note that this script first downloads all triple classification models on CoDEx-S and CoDEx-M and saves them to ```models/triple-classification/codex-{s,m}/``` if they do not already exist. 

#### <a id="baseline-script">Comparison to FB15k-237</a>

```scripts/baseline.sh``` compares a simple frequency baseline to the best model on CoDEx-M and the FB15K-237 benchmark.
The results are saved to CSV files named ```fb.csv``` and ```codex.csv```, respectively. 
To run:
```
# run from codex/
# this may take a few minutes
scripts/baseline.sh  # runs on CPU
```
Note that this script first downloads the [best pretrained LibKGE model on FB15K-237](https://github.com/uma-pi1/kge#results-and-pretrained-models) to ```models/link-prediction/fb15k-237/rescal/``` and the best link prediction model on CoDEx-M to ```models/link-prediction/codex-m/complex/``` if they do not already exist. 

### <a id="pretrained">Downloading pretrained models via the command line</a>

To **download pretrained models via the command line**, use our ```download_pretrained.py``` Python script.
The arguments are as follows:
```
usage: download_pretrained.py [-h]
                              {s,m,l} {triple-classification,link-prediction}
                              {rescal,transe,complex,conve,tucker}
                              [{rescal,transe,complex,conve,tucker} ...]

positional arguments:
  {s,m,l}               CoDEx dataset to download model(s)
  {triple-classification,link-prediction}
                        Task to download model(s) for
  {rescal,transe,complex,conve,tucker}
                        Model(s) to download for this task
```
For example, if you want to download the pretrained **link prediction** models for **ComplEx and ConvE** on **CoDEx-M**:
```
# run from codex/
python download_pretrained.py m link-prediction complex conve
```
This script will place a ```checkpoint_best.pt``` LibKGE checkpoint file in ```models/link-prediction/codex-m/complex/``` and ```models/link-prediction/codex-m/conve/```, respectively. 

Alternatively, you can download the models manually following the links we provide here. 


### <a id="lp">Link prediction results</a>

#### <a id="s-lp">CoDEx-S</a>

|  | MRR | Hits@1 | Hits@3 | Hits@10 | Config file | Pretrained model |
|---------|----:|----:|-------:|--------:|------------:|-----------------:|
| RESCAL | 0.404 | 0.293 | 0.4494 | 0.623 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/rescal/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/v209jchl93mmeuv/codex-s-lp-rescal.pt?dl=0) |
| TransE | 0.354 | 0.219 | 0.4218 | 0.634 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/transe/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/8brqhb4bd5gnktc/codex-s-lp-transe.pt?dl=0) |
| ComplEx | 0.465 | 0.372 | 0.5038 | 0.646 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/complex/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/kk3pgdnyddsdzn9/codex-s-lp-complex.pt?dl=0) |
| ConvE | 0.444 | 0.343 | 0.4926  | 0.635 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/conve/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/atvu77pzed6mcgh/codex-s-lp-conve.pt?dl=0) |
| TuckER | 0.444 | 0.339 | 0.4975 | 0.638 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-s/tucker/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/f87xloe2g3f4fvy/codex-s-lp-tucker.pt?dl=0) |

#### <a id="m-lp">CoDEx-M</a>

|  | MRR | Hits@1 | Hits@3 |Hits@10 | Config file | Pretrained model |
|---------|----:|----:|-------:|--------:|------------:|-----------------:|
| RESCAL | 0.317 | 0.244 | 0.3477 | 0.456 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/rescal/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/e3kp3eu4nnknn5b/codex-m-lp-rescal.pt?dl=0) |
| TransE | 0.303 | 0.223 | 0.3363 | 0.454 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/transe/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/y8uucaajpofct3x/codex-m-lp-transe.pt?dl=0) |
| ComplEx | 0.337 | 0.262 | 0.3701 | 0.476 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/complex/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/psy21fvbn5pbmw6/codex-m-lp-complex.pt?dl=0) |
| ConvE | 0.318 | 0.239 | 0.3551 | 0.464 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/conve/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/awjhlrfjrgz9phi/codex-m-lp-conve.pt?dl=0) |
| TuckER | 0.328 | 0.259 | 0.3599 | 0.458 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-m/tucker/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/so5l2owtx7wcos1/codex-m-lp-tucker.pt?dl=0) |

#### <a id="l-lp">CoDEx-L</a>

|  | MRR | Hits@1 | Hits@3 | Hits@10 | Config file | Pretrained model |
|---------|----:|----:|-------:|--------:|------------:|-----------------:|
| RESCAL | 0.304 | 0.242 | 0.3313 | 0.419 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/rescal/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/wvbef9u98vmkbi8/codex-l-lp-rescal.pt?dl=0) |
| TransE | 0.187 | 0.116 | 0.2188 | 0.317 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/transe/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/s9d682b49tuq5mc/codex-l-lp-transe.pt?dl=0) |
| ComplEx | 0.294 | 0.237 | 0.3179 | 0.400 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/complex/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/jqubvr77og2pvzv/codex-l-lp-complex.pt?dl=0) |
| ConvE | 0.303 | 0.240 | 0.3298 | 0.420 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/conve/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/qcfjy6i1sqbec0z/codex-l-lp-conve.pt?dl=0) |
| TuckER | 0.309 | 0.244 | 0.3395 | 0.430 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/link-prediction/codex-l/tucker/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/j8u4nqwzz3v7jw1/codex-l-lp-tucker.pt?dl=0) |


### <a id="tc">Triple classification results</a>


#### <a id="s-tc">CoDEx-S</a>

|  | Acc | F1 | Config file | Pretrained model |
|--------|----:|---:|------------:|-----------------:|
| RESCAL | 0.843 | 0.852 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-s/rescal/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/qedwhaus6gwyf1z/codex-s-tc-rescal.pt?dl=0) |
| TransE | 0.829 | 0.837 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-s/transe/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/jkp6oxcgt28ki42/codex-s-tc-transe.pt?dl=0) |
| ComplEx | 0.836 | 0.846 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-s/complex/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/2d8clm7em6ygida/codex-s-tc-complex.pt?dl=0) |
| ConvE | 0.841 | 0.846 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-s/conve/config.yaml) | [1vsAll-kl](https://www.dropbox.com/s/4rnexlf56x5qwvs/codex-s-tc-conve.pt?dl=0) |
| TuckER | 0.840 | 0.846 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-s/tucker/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/xrlfygg6ck2z3ue/codex-s-tc-tucker.pt?dl=0) |

#### <a id="m-tc">CoDEx-M</a>

|  | Acc | F1 | Config file | Pretrained model |
|--------|----:|---:|------------:|-----------------:|
| RESCAL | 0.818 | 0.815 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-m/rescal/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/366h6xwccbvqkm8/codex-m-tc-rescal.pt?dl=0) |
| TransE | 0.797 | 0.803 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-m/transe/config.yaml) | [NegSamp-kl](https://www.dropbox.com/s/0uil6mrrtadtqoe/codex-m-tc-transe.pt?dl=0) |
| ComplEx | 0.824 | 0.818 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-m/complex/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/0yh95rtgvv12qxs/codex-m-tc-complex.pt?dl=0) |
| ConvE | 0.826 | 0.829 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-m/conve/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/s9fwf1v57mm23l8/codex-m-tc-conve.pt?dl=0) |
| TuckER | 0.823 | 0.816 | [config.yaml](https://github.com/tsafavi/codex/tree/master/models/triple-classification/codex-m/tucker/config.yaml) | [KvsAll-kl](https://www.dropbox.com/s/jj09uah9cjkukl0/codex-m-tc-tucker.pt?dl=0) |


## <a id="data">Data directory structure</a>

The ```data/``` directory is structured as follows:
```
.
├── entities
│   ├── ar
│   ├── de
│   ├── en
│   ├── es
│   ├── ru
│   └── zh
├── relations
│   ├── ar
│   ├── de
│   ├── en
│   ├── es
│   ├── ru
│   └── zh
├── triples
│   ├── codex-l
│   ├── codex-m
│   ├── codex-s
│   └── raw.zip
└── types
    ├── ar
    ├── de
    ├── en
    ├── entity2types.json
    ├── es
    ├── ru
    └── zh
 ```
 
 We provide an overview of each subdirectory in this section. 
 
### <a id="entities">Entities and entity types</a>
We provide entity labels, Wikidata descriptions, and Wikipedia page extracts for entities and entity types in six languages:
Arabic (ar), German (de), English (en), Spanish (es), Russian (ru), and Chineze (zh).

Each subdirectory of ```data/entities/``` contains an ```entities.json``` file formatted as follows:
```
{
  <Wikidata entity ID>:{
    "label":<label in respective language if available>,
    "description":<Wikidata description in respective language if available>,
    "wiki":<Wikipedia page URL in respective language if available>
  }
}
```
For the labels, descriptions, or Wikipedia URLs that are not available in a given language, the value will be the empty string.

The file ```data/types/entity2types.json``` maps each Wikidata entity ID to a list of Wikidata type IDs, i.e.,
```
{
  "<Wikidata entity ID>":[
    <Wikidata type ID 1>,
    <Wikidata type ID 2>,
    ...
  ]
}
```
Each subdirectory of ```data/types/``` contains a ```types.json``` file formatted as follows: 
```
{
  <Wikidata type ID>:{
    "label":<label in respective language if available>,
    "description":<Wikidata description in respective language if available>,
    "wiki":<Wikipedia page URL in respective language if available>
  }
}
```
Each ```extracts.zip``` file contains zipped files of entity descriptions from Wikipedia.
Each file is named ```<Wikidata entity ID>.txt```. 
We provide the ```extract_en.sh``` script to unzip all English-language entity and entity type extracts.
You can edit this script and provide a different language code (```ar``` for Arabic, ```de``` for German, ```es``` for Spanish, ```ru``` for Russian, and ```zh``` for Chinese) to extract descriptions for other languages. 

### <a id="relations">Relations</a>
We provide relation labels and Wikidata descriptions for relations in six languages: 
Arabic (ar), German (de), English (en), Spanish (es), Russian (ru), and Chineze (zh).

Each subdirectory of ```data/relations/``` contains a ```relations.json``` file formatted as follows:
```
{
  <Wikidata relation ID>:{
    "label":<label in respective language if available>,
    "description":<Wikidata description in respective language if available>
  }
}
```

### <a id="triples">Triples</a>

Each triple file follows the format
```
<Wikidata head entity ID>\t<Wikidata relation ID>\t<Wikidata tail entity ID>
```
without any header or extra information per line.

If you'd like to use the raw data dump, run
```
cd data/triples
unzip raw.zip
```
This will create a new ```data/triples/raw/``` directory containing a single file, ```triples.txt```, in the same tab-separated format as the other triple files. 

# <a id="cite">How to cite</a>
You can find the full text of our paper <a href="https://arxiv.org/pdf/2009.07810.pdf" target="_blank">here</a>.

If you used our work or found it helpful, please use the following citation: 
```
@inproceedings{safavi-koutra-2020-codex,
    title = "{C}o{DE}x: A {C}omprehensive {K}nowledge {G}raph {C}ompletion {B}enchmark",
    author = "Safavi, Tara  and
      Koutra, Danai",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.669",
    doi = "10.18653/v1/2020.emnlp-main.669",
    pages = "8328--8350",
    abstract = "We present CoDEx, a set of knowledge graph completion datasets extracted from Wikidata and Wikipedia that improve upon existing knowledge graph completion benchmarks in scope and level of difficulty. In terms of scope, CoDEx comprises three knowledge graphs varying in size and structure, multilingual descriptions of entities and relations, and tens of thousands of hard negative triples that are plausible but verified to be false. To characterize CoDEx, we contribute thorough empirical analyses and benchmarking experiments. First, we analyze each CoDEx dataset in terms of logical relation patterns. Next, we report baseline link prediction and triple classification results on CoDEx for five extensively tuned embedding models. Finally, we differentiate CoDEx from the popular FB15K-237 knowledge graph completion dataset by showing that CoDEx covers more diverse and interpretable content, and is a more difficult link prediction benchmark. Data, code, and pretrained models are available at https://bit.ly/2EPbrJs.",
}
```

# <a id="ref">References and acknowledgements</a>

We thank [HeadsOfBirds](https://thenounproject.com/search/?q=lightbulb&i=1148270) for the lightbulb icon and [Evan Bond](https://thenounproject.com/icon/1113230/) for the book icon in our logo.

This project is licensed under the terms of the MIT license.

