CoDEx is a set of knowledge graph **Co**mpletion **D**atasets **Ex**tracted from Wikidata and Wikipedia. 
CoDEx offers three rich knowledge graph datasets accompanied by entity types, entity and relation descriptions, accompanying
free text taken from Wikipedia page extracts, and hard negative triples for evaluation.
We provide baseline performance results, configuration files, and pretrained models
on CoDEx using the <a href="https://github.com/uma-pi1/kge" target="_blank">LibKGE</a>
knowledge graph embedding library for two tasks, link prediction and triple classification.

## Table of contents
- <a href="#quick-start">Quick start</a>
- <a href="#data">Data</a>
    - <a href="#triples">Triples</a>
    - <a href="#entities">Entities and entity types</a>
    - <a href="#relations">Relations</a>
    - <a href="#paths">Paths</a>
- <a href="#models">Models</a>
  - <a href="#lp">Link prediction</a>
  - <a href="#tc">Triple classification</a>

## <a id="quick-start">Quick start</a>

To explore CoDEx datasets in an easy-to-use format, first install the requirements, then launch Jupyter Notebook:
```
pip install -r requirements.txt
jupyter notebook
```

## <a id="data">Data</a>


### <a id="triples">Triples</a>
Each triple file follows the format
```
<entity ID>\t<relation ID>\t<entity ID>
```
without any header or extra information per line.
The dataset statistics are as follows:

|          | Entities | Relations | Train   | Valid (+) | Test (+) | Valid (-) | Test (-) | Total triples |
|----------|----------|-----------|---------|-----------|----------|-----------|----------|---------------|
| CoDEx-S  | 2,034    | 42        | 32,888  | 1,827     | 1,828    | 1,827     | 1,828    | 36,543        |
| CoDEx-M  | 17,050   | 51        | 185,584 | 10,310    | 10,311   | 10,310    | 10,311   | 206,205       |
| CoDEx-L  | 77,951   | 69        | 551,193 | 30,622    | 30,622   | -         | -        | 612,437       |
| Raw dump | 380,038  | 75        | -       | -         | -        | -         | -        | 1,156,222     |

To unzip the raw data dump (if you plan on using it):
```
cd data/triples
unzip raw.zip
```


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

To extract all Wikipedia plain-text page excerpts for entities:
```
chmod u+x extract.sh
./extract.sh
```
This will create an ```extracts/``` folder for each language in the ```entities/``` and ```types``` directories.
Each file, named ```<Wikidata ID>.txt```, contains the excerpt for the specified Wikidata entity. 

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

### <a id="paths">Paths</a>

## <a id="models">Models</a>

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

### <a id="tc">Triple classification</a>
