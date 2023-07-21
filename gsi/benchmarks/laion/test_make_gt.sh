#!/bin/bash

#
# Config
#

# TODO: Change this to your base dataset
DATASET=/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M.npy

BASE_10K=/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_10K.npy
BASE_1M=/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_1M.npy
BASE_2M=/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_2M.npy
BASE_5M=/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_5M.npy
BASE_10M=/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_10M.npy
BASE_20M=/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_20M.npy

QUERIES=/mnt/nas1/laion400m/benchmarking/sets_nor/query_vec.npy

# TODO: Change to your desired output path
OUTPUT=/mnt/nas1/laion400m/benchmarking/sets_nor/laion24M_base_gt_cos.npy
OUTPUT_10K=/mnt/nas1/laion400m/benchmarking/sets_nor/laion24M_base_10K_gt_cos.npy
OUTPUT_1M=/mnt/nas1/laion400m/benchmarking/sets_nor/laion24M_base_1M_gt_cos.npy
OUTPUT_2M=/mnt/nas1/laion400m/benchmarking/sets_nor/laion24M_base_2M_gt_cos.npy
OUTPUT_5M=/mnt/nas1/laion400m/benchmarking/sets_nor/laion24M_base_5M_gt_cos.npy
OUTPUT_10M=/mnt/nas1/laion400m/benchmarking/sets_nor/laion24M_base_10M_gt_cos.npy
OUTPUT_20M=/mnt/nas1/laion400m/benchmarking/sets_nor/laion24M_base_20M_gt_cos.npy

#
# Script starts here
#

python make_groundtruth.py --dataset $DATASET --queries $QUERIES --o $OUTPUT --numpy 1
python make_groundtruth.py --dataset $BASE_10K --queries $QUERIES --o $OUTPUT_10K --numpy 1
python make_groundtruth.py --dataset $BASE_1M --queries $QUERIES --o $OUTPUT_1M --numpy 1
python make_groundtruth.py --dataset $BASE_2M --queries $QUERIES --o $OUTPUT_2M --numpy 1
python make_groundtruth.py --dataset $BASE_5M --queries $QUERIES --o $OUTPUT_5M --numpy 1
python make_groundtruth.py --dataset $BASE_10M --queries $QUERIES --o $OUTPUT_10M --numpy 1
python make_groundtruth.py --dataset $BASE_20M --queries $QUERIES --o $OUTPUT_20M --numpy 1
