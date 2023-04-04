#                           _       _
# __      _____  __ ___   ___  __ _| |_ ___
# \ \ /\ / / _ \/ _` \ \ / / |/ _` | __/ _ \
#  \ V  V /  __/ (_| |\ V /| | (_| | ||  __/
#   \_/\_/ \___|\__,_| \_/ |_|\__,_|\__\___|
#
#  Copyright © 2016 - 2023 Weaviate B.V. All rights reserved.
#
#  CONTACT: hello@weaviate.io
#

import os
import json
import sys
import traceback
import time
import weaviate
import requests
import argparse
import numpy as np

#
# Configuration
#

# Weaviate connection string
WEAVIATE_CONN       = "http://localhost:8091"

# Weaviate import batch size
BATCH_SIZE          = 1

# Name of the Weaviate custom class we use for benchmarking
BENCH_CLASS_NAME    = "BenchmarkDeep1B"

# File system location of all the benchmark datasets
#BENCH_DATASET_DIR   = "/mnt/nas1/fvs_benchmark_datasets/"
BENCH_DATASET_DIR   = "/Users/gwilliams/Projects/GSI/Weaviate/data"

# Set to True to print more messages for debugging purposes
VERBOSE         = True

#
# Globals
#

# The total number of imports - will be retrieved via args
TOTAL_ADDS      = -1

# Store timings for later export to CSV
STATS           = []

#
# Parse cmdline arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-n", required=True)
args = parser.parse_args()

# Set the search dabasize size
if args.n == "1M":
    TOTAL_ADDS = 1000000
else:
    TOTAL_ADDS = int(args.n)

#
# Load the GT file
#

gt_file = None
if TOTAL_ADDS == 1000000:
    gt_file = os.path.join( BENCH_DATASET_DIR, "deep-1M-gt-1000.npy" )
gt_dset = np.load(gt_file, mmap_mode='r')    
print("Got ground truth file:", gt_dset.shape)

#
# Schemaa checks
#

print("Connecting to Weaviate...")
client = weaviate.Client(WEAVIATE_CONN)
print("Done.")

print("Getting the Weaviate schema...")
schema = client.schema.get()
print(schema)
print("Done.")

# Check if the schema already has our test class.
# If so, try to delete it.
if BENCH_CLASS_NAME not in [ cls["class"] for cls in schema["classes"] ]:
    raise Exception("Could not find class '%s'" % BENCH_CLASS_NAME)
print("Found class='%s'.  Verifying schema..." % BENCH_CLASS_NAME)

# Get class schema and validate
cls_schema = None
for cls in schema["classes"]:
    if cls["class"] == BENCH_CLASS_NAME: cls_schema = cls
if cls_schema==None:
    raise Exception("Could not retrieve schema for class='%s'" % BENCH_CLASS_NAME)
if cls_schema['vectorIndexType'] != "hnsw":
    raise Exception("The schema for class='%s' is not an hnsw index." % BENCH_CLASS_NAME)
print("Verified.")

# Get object count
print("Getting db size...")
resp = client.query.aggregate(BENCH_CLASS_NAME).with_meta_count().do()
# print(resp)
# should look something like this - {'data': {'Aggregate': {'Benchmark_Deep1B': [{'meta': {'count': 0}}]}}}
count = -1
try:
    count = resp['data']['Aggregate'][BENCH_CLASS_NAME][0]['meta']['count']
except:
    traceback.print_exc()
    raise Exception("Could not get size for '%s'" % BENCH_CLASS_NAME)
print("Got database size=%d, verifying..." % count)

# Check its the right db size
if count !=  TOTAL_ADDS:
    raise Exception("Expected database size of %d but got %d" % ( TOTAL_ADDS, count ))
print("Verifed.")

#
# Perform searches
#

nearText = {"concepts": [ "q-9" ]}
result = client.query.get( BENCH_CLASS_NAME, ["index"] ).with_additional(['lastUpdateTimeUnix']).with_near_text(nearText).with_limit(10).do()
print("Response=", result)
print("GT=", gt_dset[9][0:10])

#additional_props = {
#  "classification" : ["searchTime"]
#}
#result = client.query.get( BENCH_CLASS_NAME, ["index"] ).with_additional(additional_props).with_near_text(nearText).with_limit(10).do()
#print(result)

sys.exit(0)

