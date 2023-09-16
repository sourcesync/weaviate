#!/bin/bash

set -e
set -x

#
# You need to change these parameters to reflect your local setup
#

# Write/append the benchmark results to this file
DT=$(date +%s)
DTDIR="results/$DT"
mkdir -p $DTDIR
OUTPUT="$DTDIR/benchmarks-deep50M-q10-$DT.csv"

# Set a valid allocation id
ALLOCATION_ID="0b391a1a-b916-11ed-afcb-0242ac1c0002" #"fd283b38-3e4a-11eb-a205-7085c2c5e516"

# Path to dataset numpy file
DATASET="/mnt/nas1/fvs_benchmark_datasets/deep-50M.npy"

# Path to queries numpy file
QUERIES="/mnt/nas1/fvs_benchmark_datasets/deep-queries-10.npy"

# Path to the ground truth numpy file
GROUNDTRUTH="/mnt/nas1/fvs_benchmark_datasets/deep-50M-gt-10.npy"

# Set Swagger to emit verbose http payload messages
export GEMINI_SWAGGER_VERBOSE='true'

#
# Check the arguments and the system before running the benchmarks.
#
if [ -f "$OUTPUT" ]; then
	echo "Error:  The file $OUTPUT already exists.  Please move that file or use a file path that does not already exist."
	exit 1
fi

#
# Now run all the benchmarks.  Note that this might take a while, so you should consider running it behind the 'screen' utility.
#
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 768 2>&1 | tee "$DTDIR/50M_10_768b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 768 2>&1 | tee "$DTDIR/50M_10_768b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 768 2>&1 | tee "$DTDIR/50M_10_768b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 512 2>&1 | tee "$DTDIR/50M_10_512b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 512 2>&1 | tee "$DTDIR/50M_10_512b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 512 2>&1 | tee "$DTDIR/50M_10_512b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 256 2>&1 | tee "$DTDIR/50M_10_256b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 256 2>&1 | tee "$DTDIR/50M_10_256b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 256 2>&1 | tee "$DTDIR/50M_10_256b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 128 2>&1 | tee "$DTDIR/50M_10_128b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 128 2>&1 | tee "$DTDIR/50M_10_128b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 128 2>&1 | tee "$DTDIR/50M_10_128b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 64  2>&1 | tee "$DTDIR/50M_10_64b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 64  2>&1 | tee "$DTDIR/50M_10_64b_$DT_$(date +%s).txt"
python -u gemini_fvs.py -a "$ALLOCATION_ID" -d "$DATASET" -q "$QUERIES" -g "$GROUNDTRUTH"  -o "$OUTPUT" --b 64  2>&1 | tee "$DTDIR/50M_10_64b_$DT_$(date +%s).txt"

echo "Done. $OUTPUT"

