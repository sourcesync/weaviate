#!/bin/bash

set -e
set -x

USER=gwilliams
SOURCEDIR="/mnt/nas1/weaviate_benchmark_results/algodirect"
SOURCE="$USER@192.168.99.40:$SOURCEDIR"
mkdir -p ./results
rsync -azvdO --no-owner --no-group --no-perms "$SOURCE" ./results/

