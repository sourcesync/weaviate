#!/bin/bash

set -e
set -x

##FILE="/home/public/elastic-similarity/python-training-manager-api/cache/20230829160217_clusters/clusters/clusters_25000.pkl"
#DIR="/home/public/elastic-similarity/python-training-manager-api/cache/20230829160217_clusters/clusters/"
#FILE="/home/public/elastic-similarity/python-training-manager-api/cache/20230831080827_clusters/clusters/clusters_128000.pkl"
#DIR="/home/public/elastic-similarity/python-training-manager-api/cache/20230831080827_clusters/clusters"
FILE="/home/public/elastic-similarity/python-training-manager-api/cache/20230829162324_clusters/clusters/clusters_25000.pkl"
DIR="/home/public/elastic-similarity/python-training-manager-api/cache/20230829162324_clusters/clusters"

mkdir -p $DIR
chmod ugo+rw $DIR

python picklefix.py $FILE

echo "Done."
