# codex/data/

## Table of contents
- <a href="#data">Data</a>
    - <a href="#triples">Triples</a>
    - <a href="#entities">Entities</a>
    - <a href="#types">Entity types</a>
    - <a href="#relations">Relations</a>
    - <a href="#paths">Paths</a>
- <a href="#exploration">Data exploration</a>
- <a href="#models">Models</a>
- <a href="#scripts">Evaluation scripts</a>

## <a id="data">Data</a>

CoDEx is organized as follows:

```
.
├── entities
│   ├── ar
│   ├── de
│   ├── en
│   ├── es
│   ├── ru
│   └── zh
├── extract.sh
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

To extract all Wikipedia excerpts for entities and entity types:
```
chmod u+x extract.sh
./extract.sh
```

### <a id="types">Entity types</a>

### <a id="relations">Relations</a>

### <a id="paths">Paths</a>

## <a id="exploration">Data exploration</a>

## <a id="models">Models</a>

## <a id="scripts">Evaluation scripts</a>
