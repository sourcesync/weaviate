#!/bin/bash

if [ -z "$1" ]; then
    echo "please supply a valid username like. For example:"
    echo "$0 gwilliams"
    exit 1
fi
USER=$1

set -e
set -x

# make local results dir for copies of data
mkdir -p ./results

#
# sync hnsw algo direct benchmark data
#
SOURCEDIR="/mnt/nas1/weaviate_benchmark_results/algodirect"
SOURCE="$USER@192.168.99.40:$SOURCEDIR"
rsync -azvdO --no-owner --no-group --no-perms "$SOURCE" ./results/

#
# sync latest FVS benchmark data
#
SDIR="fvs/sv7-apu12/all_09142023/"
mkdir -p "results/$SDIR"
SOURCEDIR="/mnt/nas1/weaviate_benchmark_results/$SDIR"
SOURCE="$USER@192.168.99.40:$SOURCEDIR"
rsync -azvdO --no-owner --no-group --no-perms "$SOURCE" "./results/$SDIR"

SDIR="fvs/sv7-apu11/all_09092023/"
mkdir -p "results/$SDIR"
SOURCEDIR="/mnt/nas1/weaviate_benchmark_results/$SDIR"
SOURCE="$USER@192.168.99.40:$SOURCEDIR"
rsync -azvdO --no-owner --no-group --no-perms "$SOURCE" "./results/$SDIR"

