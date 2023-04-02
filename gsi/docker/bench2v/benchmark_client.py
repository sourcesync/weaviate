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
import argparse

#
# Configuration
#

# Weaviate connection string
WEAVIATE_CONN   = "http://localhost:8081"

# Weaviate import batch size
BATCH_SIZE      = 1

# Name of the custom class for this test program
BENCH_CLASS_NAME= "Benchmark-Deep1B"

# Set to True to print more messages
VERBOSE         = True

#
# Sanity check
#

# Parse command line
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, required=True)
args = parser.parse_args()

# Set parameters
TOTAL_ADDS = args.n

print("Connecting to Weaviate...")
client = weaviate.Client(WEAVIATE_CONN)
print("Done.")

print("Getting the Weaviate schema...")
schema = client.schema.get()
print(schema)
print("Done.")

# Check if the schema already has our test class.
# If so, try to delete it.
if BENCH_CLASS_NAME in [ cls["class"] for cls in schema["classes"] ]:
    print("Warning: Found class='%s'" % BENCH_CLASS_NAME)

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
    sys.exit(0)

# The schema class to create.
# define the class
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

# Retrieve updated schema and check its a gemini index
print("Done.  Verifying schema and hnsw index...")
schema = client.schema.get()
if BENCH_CLASS_NAME not in [ cls["class"] for cls in schema["classes"] ]:
    raise Exception("Could not verify class='%s' was created." % BENCH_CLASS_NAME)
cls_schema = None
for cls in schema["classes"]:
    if cls["class"] == BENCH_CLASS_NAME: cls_schema = cls
if cls_schema==None:
    raise Exception("Could not retrieve schema for class='%s'" % BENCH_CLASS_NAME)
if cls_schema['vectorIndexType'] != "gemini":
    raise Exception("The schema for class='%s' is not a gemini index." % BENCH_CLASS_NAME)
print("Verified.")

# Prepare a batch process for importing data to Weaviate.
print("Import documents to Weaviate (max of around %d docs)" % MAX_ADDS)

# Prepare a batch process for sending data to weaviate
print("Uploading benchmark indices to Weaviate (max of around %d strings)" % MAX_ADDS)
count = 0
while True: # lets loop until we exceed the MAX configured above
    with client.batch as batch:
        batch.batch_size=1
        # Batch import all Questions
        for i, d in enumerate(range(MAX_ADDS)):
            if VERBOSE: print(f"importing index: {i}")

            properties = {
                "index": str(i)
            }

            resp = client.batch.add_data_object(properties, "Benchmark")
            print(resp)
            count += 1
            
    if VERBOSE: print("Batch uploaded %d strings so far..." % count)
    if count == MAX_ADDS:
        break

print("Done adding %d total strings to Weaviate.  Verifying count at Weaviate..." % count)

resp = client.query.aggregate(BENCH_CLASS_NAME).with_meta_count().do()
print(resp)
sys.exit(0)

