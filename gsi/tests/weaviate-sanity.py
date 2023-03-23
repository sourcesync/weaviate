import json
import sys
import traceback
import time
import weaviate

#
# Configuration
#

# Maximum number of documents to add
MAX_ADDS = 4001

# The weaviate client with connection info
client = weaviate.Client("http://localhost:8081")  # Replace with your endpoint


# In case the weaviate class already exists from previous test, let's try to delete it here.
try:
    # delete class "YourClassName" - THIS WILL DELETE ALL DATA IN THIS CLASS
    client.schema.delete_class("Question")  # Replace with your class name
except:
    traceback.print_exc()

schema = client.schema.get()
print("Updated schema:", json.dumps(schema, indent=4))

# We will create this class "Question"
class_obj = {
    "class": "Question",
    "description": "Information from a Jeopardy! question",  # description of the class
    "properties": [
        {
            "dataType": ["text"],
            "description": "The question",
            "name": "question",
        },
        {
            "dataType": ["text"],
            "description": "The answer",
            "name": "answer",
        },
        {
            "dataType": ["string"],
            "description": "The category",
            "name": "category",
        },
    ]
}

# create the class at weaviate
client.schema.create_class(class_obj)

# print the schema
schema = client.schema.get()
print("Updated schema: ", json.dumps(schema, indent=4))

# Load the NLP sample data
import requests
url = 'https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json'
resp = requests.get(url)
data = json.loads(resp.text)
print( "data size", len(data) )

# Prepare a batch process for sending data to weaviate
count = 0
while True: # lets loop until we exceed the MAX configured above
    with client.batch as batch:
        batch.batch_size=100
        # Batch import all Questions
        for i, d in enumerate(data):
            print(f"importing question: {i+1}")

            properties = {
                "answer": d["Answer"],
                "question": d["Question"],
                "category": d["Category"],
            }

            resp = client.batch.add_data_object(properties, "Question")
            count += 1
            
    if count > MAX_ADDS:
        break
    print("Uploading %d documents so far..." % count)

print("Done adding documents.  Let's prepare to search...")
time.sleep(5)


def parse_result(result):
    '''Parse a query result into something actionable.'''
  
    async_try_again = False
    errors = []
    data = None

    #print("RESULT->", result)

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


nearText = {"concepts": ["biology"]}
consec_errs = 0
st_time = time.time()

# Now let's perform a bunch of searches
while True:

    print("Sending a search request now...")
    result = client.query.get("Question", ["question", "answer", "category"] ).with_near_text(nearText).with_limit(2).do()
    async_try_again, errors, data = parse_result(result)

    #print(async_try_again, errors, data)
    if async_try_again:
        print("FVS is building an index.  It's asked us to try the search again later...")
        time.sleep(2)
        continue
    elif errors:
        print("We got errors->", errors)
        consec_errs += 1
        if consec_errs > 5: 
            print("Too many errors.  Let's stop.")
            break
    elif data:
        print("We Got data->", data)
        consec_errs = 0
        break
    else:
        print("Unknown result! Let's stop")
        break

e_time = time.time()
print("Total time building index and performing search=", e_time-st_time)

print("Done.")
