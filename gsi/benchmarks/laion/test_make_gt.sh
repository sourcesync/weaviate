#!/bin/bash

#
# Config
#

# TODO: Change this to your base dataset
DATASET=/mnt/nas1/atlas_data/benchmarking/sets/base_atlas.npy

#DATASET=/mnt/nas1/atlas_data/benchmarking/sets/base_atlas.npy
BASE_ATLAS_1M=/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_1M.npy
BASE_ATLAS_2M=/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_2M.npy
BASE_ATLAS_5M=/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_5M.npy
BASE_ATLAS_10M=/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_10M.npy
BASE_ATLAS_20M=/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_20M.npy
BASE_ATLAS_30M=/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_30M.npy

QUERIES=/mnt/nas1/atlas_data/benchmarking/sets/query_vec.npy

# TODO: Change to your desired output path
OUTPUT=/mnt/nas1/atlas_data/benchmarking/sets_nor/atlas_base_gt_cos.npy
OUTPUT_1M=/mnt/nas1/atlas_data/benchmarking/sets_nor/atlas_base_1M_gt_cos.npy
OUTPUT_2M=/mnt/nas1/atlas_data/benchmarking/sets_nor/atlas_base_2M_gt_cos.npy
OUTPUT_5M=/mnt/nas1/atlas_data/benchmarking/sets_nor/atlas_base_5M_gt_cos.npy
OUTPUT_10M=/mnt/nas1/atlas_data/benchmarking/sets_nor/atlas_base_10M_gt_cos.npy
OUTPUT_20M=/mnt/nas1/atlas_data/benchmarking/sets_nor/atlas_base_20M_gt_cos.npy
OUTPUT_30M=/mnt/nas1/atlas_data/benchmarking/sets_nor/atlas_base_30M_gt_cos.npy

#
# Script starts here
#

python make_groundtruth.py --dataset $DATASET --queries $QUERIES --o $OUTPUT --numpy 1
python make_groundtruth.py --dataset $BASE_ATLAS_1M --queries $QUERIES --o $OUTPUT_1M --numpy 1
python make_groundtruth.py --dataset $BASE_ATLAS_2M --queries $QUERIES --o $OUTPUT_2M --numpy 1
python make_groundtruth.py --dataset $BASE_ATLAS_5M --queries $QUERIES --o $OUTPUT_5M --numpy 1
python make_groundtruth.py --dataset $BASE_ATLAS_10M --queries $QUERIES --o $OUTPUT_10M --numpy 1
python make_groundtruth.py --dataset $BASE_ATLAS_20M --queries $QUERIES --o $OUTPUT_20M --numpy 1
python make_groundtruth.py --dataset $BASE_ATLAS_30M --queries $QUERIES --o $OUTPUT_30M --numpy 1
