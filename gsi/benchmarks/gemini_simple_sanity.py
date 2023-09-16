import os
import json
import sys
import traceback
import time
import weaviate
import requests
import argparse
import pandas as pd
import platform
import shutil
import socket
import numpy
import swagger_client
import uuid
from swagger_client.models import *


#
# configuration ( you should not change these unless you know what you are doing! )
#
gemini_parms    = {'skip': False, 'searchType': 'flat', 'centroidsHammingK': 5000, 'centroidsRerank': 4000, 'hammingK': 3200, 'nBits': 768, "filePath":"" }

allocation_id   = 'fvs-automation'

weaviate_conn   = "http://localhost:8091"

class_name      = "BenchmarkDeep1B"



#
# load datasets
#
dset = numpy.load("/mnt/nas1/fvs_benchmark_datasets/deep-10K.npy")
print("loaded dset", dset)


#
# connect to weaviate and get the schema
#
client = weaviate.Client(weaviate_conn)
print("client=",client)

schema = client.schema.get()
print("schema=",schema)


#
# do some light schema checks
#

if class_name in [ cls["class"] for cls in schema["classes"] ]:
    raise Exception("This sanity check does not work if %s is already present at weaviaate.  Please delete it, or weaviate's database files." %  class_name)


#
# create the class
#
class_obj = {
    "class": class_name,
    "description": "The sanity dataset class for gemini index.",
    "properties": [
        {
            "dataType": ["text"],
            "description": "The array index as a string",
            "name": "index",
        }
    ],
    "vectorIndexType": "gemini",
    "vectorIndexConfig": gemini_parms
}
client.schema.create_class(class_obj)
print("class created")

schema = client.schema.get()
print("new schema=", schema)


#
# Add data
#
print("adding data...")
with client.batch as batch:
    
    batch.batch_size=1 # gemini plugin only supports bs=1

    for d in range(dset.shape[0]):

        if (d % 1000) ==0: # confirm count at every 1000

            resp = client.query.aggregate(class_name).with_meta_count().do()
            print(resp)
            interim_count = resp['data']['Aggregate'][class_name][0]['meta']['count']

            if interim_count != d:
                raise Exception("Unexpected object count=%d, expected=%d" % ( interim_count, d))


        # add the d'th vector
        vector = dset[d]
        resp = client.batch.add_data_object({
                'counter': d
            },
            class_name,
            str(uuid.uuid3(uuid.NAMESPACE_DNS, str(d))),
            vector = vector
        )
        if 'error' in resp:
            raise Exception("Add failed at item=%d" % d)

        


#
# confirm final expected count
#

resp = client.query.aggregate(class_name).with_meta_count().do()
count = resp['data']['Aggregate'][class_name][0]['meta']['count']
if count != dset.shape[0]:
    raise Exception("Unexpected object count=%d " % ( count) )

print("done adding vectors")


#
# loop on search call until training finished, or an error
#

def parse_search_result(result):
    '''Parse a query result into something actionable.'''

    async_try_again = False
    errors = []
    data = None

    # First loop through errors if any.  
    # We look for "gemini async build" messages 
    # and don't interpret them as errors.
    if "errors" in result.keys():
        errs = result["errors"]
        for err in errs:
            if "message" in err.keys():
                mesg = err["message"]
                if mesg.find("vector search: Async index build is in progress.")>=0:
                    async_try_again = True
                elif mesg.find("vector search: Async index build completed.")>=0:
                    async_try_again = True
                else:
                    errors.append(err)

    elif "data" in result.keys():
        data = result["data"]

    return async_try_again, errors, data

print("initiating training...")
consec_errs = 0
while True:

    # make a search call
    nearVector = { "vector": dset[0].tolist() } # use a dummy vector
    result = client.query.get(class_name, ["counter"]).with_near_vector(nearVector).with_limit(10).do()

    # Interpret the search call results
    async_try_again, errors, data = parse_search_result(result)
    if async_try_again:
        print("gemini is still asynchronously building an index, and has asked us to try the search again a little later...")
        time.sleep(2)
        continue
    elif errors:
        print("We got search actual errors->", errors)
        consec_errs += 1
        if consec_errs > 5: raise Exceptions("Too many errors.  Let's stop here.")
    elif data:
        print("Successful search, data->", data)
        break
    else:
        raise Exception("Unknown result! Let's stop here.")


print("Gemini index done training and its loaded into the APU(s), you can now perform vector searches on %s" % class_name)


