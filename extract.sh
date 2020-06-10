#!/bin/bash
set -e
code=$1
unzip data/entities/${code}/extracts.zip -d data/entities/${code}/
unzip data/types/${code}/extracts.zip -d data/types/${code}/
