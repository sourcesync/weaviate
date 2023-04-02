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

import json
import sys
import traceback
import time
import weaviate
import requests
import argparse

#
# Configuration
#

# Weaviate connection string
WEAVIATE_CONN   = "http://localhost:8091"

# Weaviate import batch size
BATCH_SIZE      = 1

# Name of the custom class for this test program
BENCH_CLASS_NAME= "BenchmarkDeep1B"

# Set to True to print more messages
VERBOSE         = True

#
# Sanity check
#

# Parse command line
#parser = argparse.ArgumentParser()
#args = parser.parse_args()

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


print("Warning: Found class='%s'.  Verifying..." % BENCH_CLASS_NAME)

# Get class schema and validate
cls_schema = None
for cls in schema["classes"]:
    if cls["class"] == BENCH_CLASS_NAME: cls_schema = cls
if cls_schema==None:
    raise Exception("Could not retrieve schema for class='%s'" % BENCH_CLASS_NAME)
if cls_schema['vectorIndexType'] != "hnsw":
    raise Exception("The schema for class='%s' is not an hnsw index." % BENCH_CLASS_NAME)

# Get object count
resp = client.query.aggregate(BENCH_CLASS_NAME).with_meta_count().do()
print(resp)
# should look something like this - {'data': {'Aggregate': {'Benchmark_Deep1B': [{'meta': {'count': 0}}]}}}
count = 0
try:
    count = resp['data']['Aggregate'][BENCH_CLASS_NAME][0]['meta']['count']
except:
    traceback.print_exc()
    raise Exception("Could not get count for '%s'" % BENCH_CLASS_NAME)
print("Got count=", count)

sys.exit(0)

