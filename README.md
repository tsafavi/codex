CoDEx is a set of knowledge graph **Co**mpletion **D**atasets **Ex**tracted from Wikidata and Wikipedia. 
CoDEx offers three rich knowledge graph datasets that contain positive and hard negative triples, entity types, entity and relation descriptions, and Wikipedia page extracts for entities. 
We provide baseline performance results, configuration files, and pretrained models
on CoDEx using the <a href="https://github.com/uma-pi1/kge" target="_blank">LibKGE</a> library for two tasks, link prediction and triple classification.

## Table of contents
- <a href="#quick-start">Quick start</a>
- <a href="#data">Data</a>
    - <a href="#triples">Triples</a>
    - <a href="#entities">Entities and entity types</a>
    - <a href="#relations">Relations</a>
- <a href="#paths">Paths</a>
- <a href="#models">Pretrained models and results</a>
  - <a href="#lp">Link prediction</a>
    - <a href="#s-lp">CoDEx-S</a>
    - <a href="#m-lp">CoDEx-M</a>
    - <a href="#l-lp">CoDEx-L</a>
  - <a href="#tc">Triple classification</a>
    - <a href="#s-tc">CoDEx-S</a>
    - <a href="#m-tc">CoDEx-M</a>
- <a href="#scripts">Evaluation scripts</a>
  - <a href="#tc-script">Triple classification</a>
  - <a href="#baseline-script">Link prediction baseline comparison</a>

## <a id="quick-start">Quick start</a>


To explore the CoDEx datasets in an easy-to-use interface, 
first clone the repository:
```
git clone <git_url_here/codex.git>
cd codex
```
Extract all Wikipedia plain-text page excerpts for entities:
```
chmod u+x extract.sh
./extract.sh
```
Finally, install the Python requirements, then launch Jupyter Notebook:
```
pip install -r requirements.txt
jupyter notebook
```
You should now be able to open the ```Explore CoDEx.ipynb``` notebook in your browser, which provides a glimpse into how the datasets are structured and what kinds of information you can obtain from each dataset. 

## <a id="data">Data</a>


### <a id="triples">Triples</a>

The dataset statistics are as follows:

|          | Entities | Relations | Train   | Valid (+) | Test (+) | Valid (-) | Test (-) | Total triples |
|----------|---------:|----------:|--------:|----------:|---------:|----------:|---------:|--------------:|
| CoDEx-S  | 2,034    | 42        | 32,888  | 1,827     | 1,828    | 1,827     | 1,828    | 36,543        |
| CoDEx-M  | 17,050   | 51        | 185,584 | 10,310    | 10,311   | 10,310    | 10,311   | 206,205       |
| CoDEx-L  | 77,951   | 69        | 551,193 | 30,622    | 30,622   | -         | -        | 612,437       |
| Raw dump | 380,038  | 75        | -       | -         | -        | -         | -        | 1,156,222     |

Each triple file follows the format
```
<entity ID>\t<relation ID>\t<entity ID>
```
without any header or extra information per line.

To unzip the raw data dump, run:
```
cd data/triples
unzip raw.zip
```
This will create a new ```data/triples/raw``` directory with a single file, ```triples.txt```, in the same tab-separated format as the other triple files. 


### <a id="entities">Entities and entity types</a>
We provide entity labels, Wikidata descriptions, and Wikipedia page extracts for entities and entity types in six languages:
Arabic (ar), German (de), English (en), Spanish (es), Russian (ru), and Chineze (zh).
Each subdirectory of ```entities/``` and ```types``` contains an ```entities.json``` file formatted as follows:
```
{
  <Wikidata entity ID>:{
    "label":<label in respective language if available>,
    "description":<Wikidata description in respective language if available>,
    "wiki":<Wikipedia page URL in respective language if available>
  }
}
```

The file ```types/entity2types.json``` maps each Wikidata entity ID to a list of Wikidata type IDs, i.e.,
```
{
  "<Wikidata entity ID>":[
    <Wikidata type ID 1>,
    <Wikidata type ID 2>,
    ...
  ]
}
```

### <a id="relations">Relations</a>
We provide relation labels and Wikidata descriptions for relations in six languages: 
Arabic (ar), German (de), English (en), Spanish (es), Russian (ru), and Chineze (zh).
Each language directory contains an ```relations.json``` file formatted as follows:
```
{
  <Wikidata relation ID>:{
    "label":<label in respective language if available>,
    "description":<Wikidata description in respective language if available>
  }
}
```

## <a id="paths">Paths</a>
We provide compositional (multi-hop) paths of lengths two and three, discovered using <a href="https://github.com/lajus/amie" target="_blank">AMIE 3</a>, on each CoDEx dataset in the ```paths``` directory. 
Each set of paths is provided as a TSV file.
The ```Rule``` column gives paths in the following format: 
```
?var1 <relation ID 1> ?var2 ?var2 <relation ID 2> ?var3 => ?var1 <relation ID 3> ?var3
```
To understand the other outputs of AMIE 3, take a look at:
> Jonathan Lajus, Luis Galárraga, Fabian M. Suchanek </br>
> <a href="https://suchanek.name/work/publications/eswc-2020-amie-3.pdf" target="_blank">Fast and Exact Rule Mining with AMIE 3</a>  </br>
> Extended Semantic Web Conference (ESWC), 2020

We also provide an overview of the compositional paths in CoDEx in the quick-start exploration notebook. 

## <a id="models">Pretrained models and results</a>

To use the pretrained models or run any scripts that involve pretrained models, you will need to install LibKGE by
<a href="https://github.com/uma-pi1/kge#quick-start" target="_blank">following the installation instructions</a>.
After installing LibKGE, take note of the path to your local LibKGE installation, and run the following:
```
chmod u+x libkge_setup.sh
./libkge_setup <your_local_path_to_libkge>
```
This script will copy each CoDEx dataset to LibKGE's ```data/``` directory and preprocess each dataset according to
the format the LibKGE requires. 

### <a id="lp">Link prediction</a>

#### <a id="s-lp">CoDEx-S</a>

|  | MRR | Hits@1 | Hits@10 | Config file | Pretrained model |
|---------|----:|-------:|--------:|------------:|-----------------:|
| RESCAL | 0.399 | 0.289 | 0.625 | <a href="models/link-prediction/codex-s/rescal/config.yaml">config.yaml</a> |  |
| TransE | 0.353 | 0.216 | 0.624 | <a href="models/link-prediction/codex-s/transe/config.yaml">config.yaml</a> |  |
| ComplEx | 0.456 | 0.372 | 0.643 | <a href="models/link-prediction/codex-s/complex/config.yaml">config.yaml</a> |  |
| ConvE | 0.449 | 0.353 | 0.636 | <a href="models/link-prediction/codex-s/conve/config.yaml">config.yaml</a> |  |

#### <a id="m-lp">CoDEx-M</a>

|  | MRR | Hits@1 | Hits@10 | Config file | Pretrained model |
|---------|----:|-------:|--------:|------------:|-----------------:|
| RESCAL | 0.317 | 0.244 | 0.456 | <a href="models/link-prediction/codex-m/rescal/config.yaml">config.yaml</a> |  |
| TransE | 0.303 | 0.223 | 0.454 | <a href="models/link-prediction/codex-m/transe/config.yaml">config.yaml</a> |  |
| ComplEx | 0.337 | 0.262 | 0.476 | <a href="models/link-prediction/codex-m/complex/config.yaml">config.yaml</a> |  |
| ConvE | 0.318 | 0.239 | 0.464 | <a href="models/link-prediction/codex-m/conve/config.yaml">config.yaml</a> |  |

#### <a id="l-lp">CoDEx-L</a>

|  | MRR | Hits@1 | Hits@10 | Config file | Pretrained model |
|---------|----:|-------:|--------:|------------:|-----------------:|
| RESCAL | 0.304 | 0.242 | 0.419 | <a href="models/link-prediction/codex-l/rescal/config.yaml">config.yaml</a> |  |
| TransE | 0.187 | 0.116 | 0.317 | <a href="models/link-prediction/codex-l/transe/config.yaml">config.yaml</a> |  |
| ComplEx | 0.294 | 0.237 | 0.400 | <a href="models/link-prediction/codex-l/complex/config.yaml">config.yaml</a> |  |
| ConvE | 0.303 | 0.240 | 0.420 | <a href="models/link-prediction/codex-l/conve/config.yaml">config.yaml</a> |  |

### <a id="tc">Triple classification</a>

#### <a id="s-tc">CoDEx-S</a>

|  | Acc | F1 | Config file | Pretrained model |
|--------|----:|---:|------------:|-----------------:|
| RESCAL | 0.804 | 0.800 | <a href="models/triple-classification/codex-s/rescal/config.yaml">config.yaml</a> |  |
| TransE | 0.659 | 0.632 | <a href="models/triple-classification/codex-s/transe/config.yaml">config.yaml</a> |  |
| ComplEx | 0.810 | 0.808 | <a href="models/triple-classification/codex-s/complex/config.yaml">config.yaml</a> |  |
| ConvE | 0.734 | 0.744 | <a href="models/triple-classification/codex-s/conve/config.yaml">config.yaml</a> |  |

#### <a id="m-tc">CoDEx-M</a>

|  | Acc | F1 | Config file | Pretrained model |
|--------|----:|---:|------------:|-----------------:|
| RESCAL |  |  | <a href="models/triple-classification/codex-m/rescal/config.yaml">config.yaml</a> |  |
| TransE |  |  | <a href="models/triple-classification/codex-m/transe/config.yaml">config.yaml</a> |  |
| ComplEx | 0.765 | 0.751 | <a href="models/triple-classification/codex-m/complex/config.yaml">config.yaml</a> |  |
| ConvE |  |  | <a href="models/triple-classification/codex-m/conve/config.yaml">config.yaml</a> |  |


### <a id="scripts">Evaluation scripts</a>

We provide several evaluation scripts to reproduce results in our paper.
**Please note that these scripts rely on LibKGE, so if you update your installation of LibKGE and the API changes, you may need to update the scripts accordingly.**

#### <a id="tc-script">Triple classification</a>

```
chmod u+x scripts/tc.sh
scripts/tc.sh
```
This script runs triple classification on each model in the ```models/triple-classification/``` directory for each dataset and outputs validation and test accuracy/F1. 

#### <a id="baseline-script">Link prediction baseline comparison</a>

```
chmod u+x scripts/baseline.sh
scripts/baseline.sh
```
This script downloads the <a href="https://github.com/uma-pi1/kge#results-and-pretrained-models" target="_blank">best pretrained LibKGE model on FB15K-237</a> to the ```models/link-prediction/``` directory, then compares a simple frequency baseline to the best model on FB15K-237 and CoDEx-M, saving the results for each dataset to a CSV.
