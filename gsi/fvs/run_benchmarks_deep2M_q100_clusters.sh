#!/bin/bash

#
# You need to change these parameters to reflect your local setup
#

# Write/append the benchmark results to this file
OUTPUT="results/benchmarks-deep2M-q100-clusters.csv"

# Set a valid allocation id
ALLOCATION_ID="0b391a1a-b916-11ed-afcb-0242ac1c0002" #"fd283b38-3e4a-11eb-a205-7085c2c5e516"

# Path to dataset numpy file
DATASET="/mnt/nas1/fvs_benchmark_datasets/deep-2M.npy"

# Path to queries numpy file
QUERIES="/mnt/nas1/fvs_benchmark_datasets/deep-queries-100.npy"

# Path to the ground truth numpy file
GROUNDTRUTH="/mnt/nas1/fvs_benchmark_datasets/deep-2M-gt-100.npy"

# Set Swagger to emit verbose http payload messages
export GEMINI_SWAGGER_VERBOSE='true'

#
# Check the arguments and the system before running the benchmarks.
#
if [ -f "$OUTPUT" ]; then
	echo "Error:  The file $OUTPUT already exists.  Please move that file or use a file path that does not already exist."
	exit 1
fi

# Make sure this script exists on any error going forward.
set -e

#
# Now run all the benchmarks.  Note that this might take a while, so you should consider running it behind the 'screen' utility.
#
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 768 2>&1 | tee "./results/2M_100_768b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 768 2>&1 | tee "./results/2M_100_768b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 768 2>&1 | tee "./results/2M_100_768b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 512 2>&1 | tee "./results/2M_100_512b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 512 2>&1 | tee "./results/2M_100_512b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 512 2>&1 | tee "./results/2M_100_512b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 256 2>&1 | tee "./results/2M_100_256b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 256 2>&1 | tee "./results/2M_100_256b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 256 2>&1 | tee "./results/2M_100_256b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 128 2>&1 | tee "./results/2M_100_128b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 128 2>&1 | tee "./results/2M_100_128b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 128 2>&1 | tee "./results/2M_100_128b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 64 2>&1 | tee "./results/2M_100_64b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 64 2>&1 | tee "./results/2M_100_64b_$(date +%s).txt"
python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 64 2>&1 | tee "./results/2M_100_64b_$(date +%s).txt"


