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
import pandas as pd

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
VERBOSE         = False

# Timing each import call
BENCHMARK_DETAILED  = False

#
# Globals
#

# The total number of imports - will be retrieved via args
TOTAL_ADDS      = -1 

# Store timings for later export to CSV
STATS           = []

#
# Parse cmd line arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, required=True)
args = parser.parse_args()

#
# Start schema checks and import
#

# Set number items to import
TOTAL_ADDS = args.n

print("Connecting to Weaviate...")
client = weaviate.Client(WEAVIATE_CONN)
print("Done.")

print("Getting the Weaviate schema...")
schema = client.schema.get()
print(schema)
print("Done.")

# Schema check...
STATS.append( {"event": "start_schema_check", "ts": time.time()} )
if BENCH_CLASS_NAME in [ cls["class"] for cls in schema["classes"] ]:
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
    if count != 0:
        raise Exception("Unexpected object count (%d) for '%s'" % ( count, BENCH_CLASS_NAME ))
    print("Verified.")

else:
    # The schema class to create. Define the class first and then set..."

    class_obj = {
        "class": BENCH_CLASS_NAME,
        "description": "The benchmark dataset class for Deep1B",  
        "properties": [
            {
                "dataType": ["text"],
                "description": "The array index as a string",
                "name": "index",
            }
        ],
        "vectorIndexType": "hnsw"
    }

    # Update the schema with this class
    print("Creating '%s' with hnsw index..." % BENCH_CLASS_NAME)
    client.schema.create_class(class_obj)

    # Retrieve updated schema and check it...
    print("Done.  Verifying schema and hnsw index...")
    schema = client.schema.get()
    if BENCH_CLASS_NAME not in [ cls["class"] for cls in schema["classes"] ]:
        raise Exception("Could not verify class='%s' was created." % BENCH_CLASS_NAME)
    cls_schema = None
    for cls in schema["classes"]:
        if cls["class"] == BENCH_CLASS_NAME: cls_schema = cls
    if cls_schema==None:
        raise Exception("Could not retrieve schema for class='%s'" % BENCH_CLASS_NAME)
    if cls_schema['vectorIndexType'] != "hnsw":
        raise Exception("The schema for class='%s' is not a hnsw index." % BENCH_CLASS_NAME)
    print("Verified.")
STATS.append( {"event": "end_schema_check", "ts": time.time()} )

# Prepare a batch process for importing data to Weaviate.
print("Import documents to Weaviate (max of %d docs)" % TOTAL_ADDS)

# Prepare a batch process for sending data to weaviate
print("Uploading benchmark indices to Weaviate (max of around %d strings)" % TOTAL_ADDS)
count = 0
while True: # lets loop until we exceed the MAX configured above
    with client.batch as batch:
        batch.batch_size=1
        # Batch import all Questions
        for i, d in enumerate(range(TOTAL_ADDS)):
            if VERBOSE: print(f"importing index: {i}")

            properties = {
                "index": str(i)
            }

            if BENCHMARK_DETAILED: STATS.append( {"event": "adding %d/%d" % ((i+1), TOTAL_ADDS), "ts": time.time()} )
            elif (i % 1000) ==0: STATS.append( {"event": "adding %d/%d" % ((i+1), TOTAL_ADDS), "ts": time.time()} )

            resp = client.batch.add_data_object(properties, BENCH_CLASS_NAME)
            if 'error' in resp:
                print("Got error adding object->", resp)
                raise Exception("Add failed at %d" % idx)

            if BENCHMARK_DETAILED: STATS.append( {"event": "added %d/%d" % ((i+1),TOTAL_ADDS), "ts": time.time()} )
            elif (i % 1000) ==0: STATS.append( {"event": "added %d/%d" % ((i+1), TOTAL_ADDS), "ts": time.time()} )

            if (i % 1000)==0: print("Imported %d/%d so far..." % (i+1, TOTAL_ADDS) )

            count += 1
            
    if VERBOSE: print("Batch uploaded %d strings so far..." % count)
    if count == TOTAL_ADDS:
        break

STATS.append( {"event": "done all %d" % count, "ts": time.time()} )
print("Done adding %d total strings to Weaviate.  Verifying count at Weaviate..." % count)

resp = client.query.aggregate(BENCH_CLASS_NAME).with_meta_count().do()
count = 0
try:
    count = resp['data']['Aggregate'][BENCH_CLASS_NAME][0]['meta']['count']
except:
    traceback.print_exc()
    raise Exception("Could not get count for '%s'" % BENCH_CLASS_NAME)
STATS.append( {"event": "import verified %d" % count, "ts": time.time()} )
if count != TOTAL_ADDS:
    raise Exception("Unexpected object count (%d) for '%s'" % ( count, BENCH_CLASS_NAME ))
print("Verified.")

# export the STATS csv
df = pd.DataFrame(STATS)
fname = "results/%s__%d__%f.csv" % ( BENCH_CLASS_NAME, count, time.time() )
df.to_csv(fname)
print("Wrote", fname)

sys.exit(0)
