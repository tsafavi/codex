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
    - <a href="#entities">Entities</a>
    - <a href="#types">Entity types</a>
    - <a href="#relations">Relations</a>
    - <a href="#paths">Paths</a>
- <a href="#exploration">Data exploration</a>
- <a href="#models">Models</a>
  - <a href="#lp">Link prediction</a>
  - <a href="#tc">Triple classification</a>
- <a href="#scripts">Evaluation scripts</a>

## <a id="quick-start">Quick start</a>

Extract all Wikipedia page excerpts for entities:
```
chmod u+x extract.sh
./extract.sh
```

Unzip the raw data dump (if you plan on using it):
```
cd data/triples
unzip raw.zip
cd ../../
```

To use the pretrained models or run any scripts that involve pretrained models, you will need to install LibKGE by
<a href="https://github.com/uma-pi1/kge#quick-start" target="_blank">following the installation instructions</a>.
After installing LibKGE, take note of the path to your local LibKGE installation, and run the following:
```
chmod u+x libkge_setup.sh
./libkge_setup <your_local_path_to_libkge>
```
This script will copy each CoDEx dataset to LibKGE's ```data/``` directory and preprocess each dataset according to
the format the LibKGE requires. 

## <a id="data">Data</a>

CoDEx datasets are organized as follows:

```
.
├── entities
│   ├── ar
│   ├── de
│   ├── en
│   ├── es
│   ├── ru
│   └── zh
├── paths
│   ├── codex-l.tsv
│   ├── codex-m.tsv
│   └── codex-s.tsv
├── relations
│   ├── ar
│   ├── de
│   ├── en
│   ├── es
│   ├── ru
│   └── zh
├── test.sh
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

### <a id="triples">Triples</a>
Each triple file follows the format
```
<entity ID>\t<relation ID>\t<entity ID>
```
without any header or extra information per line.

|          | Entities | Relations | Train   | Valid (+) | Test (+) | Valid (-) | Test (-) | Total triples |
|----------|----------|-----------|---------|-----------|----------|-----------|----------|---------------|
| CoDEx-S  | 2,034    | 42        | 32,888  | 1,827     | 1,828    | 1,827     | 1,828    | 36,543        |
| CoDEx-M  | 17,050   | 51        | 185,584 | 10,310    | 10,311   | 10,310    | 10,311   | 206,205       |
| CoDEx-L  | 77,951   | 69        | 551,193 | 30,622    | 30,622   | -         | -        | 612,437       |
| Raw dump | 380,038  | 75        | -       | -         | -        | -         | -        | 1,156,222     |


### <a id="entities">Entities</a>


### <a id="types">Entity types</a>

### <a id="relations">Relations</a>

### <a id="paths">Paths</a>

## <a id="exploration">Data exploration</a>

## <a id="models">Models</a>

### <a id="lp">Link prediction</a>

### <a id="tc">Triple classification</a>

## <a id="scripts">Evaluation scripts</a>
