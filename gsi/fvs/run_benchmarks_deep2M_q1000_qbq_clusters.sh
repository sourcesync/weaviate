#!/bin/bash

set -e
set -x

#
# You need to change these parameters to reflect your local setup
#

# Write/append the benchmark results to this file
DT=$(date +%s)
DTDIR="results/qbq"
mkdir -p "$DTDIR/logs"

# Set a valid allocation id
ALLOCATION_ID="fvs-automation" 

# Path to dataset numpy file
DATASET="/mnt/nas1/fvs_benchmark_datasets/deep-2M.npy"

# Path to queries numpy file
QUERIES="/mnt/nas1/fvs_benchmark_datasets/deep-queries-1000.npy"

# Path to the ground truth numpy file
GROUNDTRUTH="/mnt/nas1/fvs_benchmark_datasets/deep-2M-gt-1000.npy"

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
function run_benchmark() {
    BITS=$1
    BASE="benchmarks-deep2M-q1000-qbq-clusters-$BITS"
    OUTPUT="$DTDIR/$BASE-$DT.csv"
    TEEOUTPUT="$DTDIR/logs/$BASE-$DT.txt"
    python -u gemini_fvs_clusters.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b "$BITS" --qbq 2>&1 | tee "$TEEOUTPUT"
}

run_benchmark 768
run_benchmark 512
run_benchmark 256
run_benchmark 128
run_benchmark 64

echo "Done. $OUTPUT"
exit
