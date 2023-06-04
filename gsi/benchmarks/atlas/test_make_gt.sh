#!/bin/bash

#
# Config
#

# TODO: Change this to your base dataset
DATASET=/mnt/nas1/atlas_data/benchmarking/atlas.npy.test

QUERIES=/mnt/nas1/atlas_data/benchmarking/query_vec.npy

# TODO: Change to your desires output path
OUTPUT="/tmp/out.npy"

#
# Script starts here
#

python make_groundtruth.py --dataset $DATASET --queries $QUERIES --o $OUTPUT --numpy 1
