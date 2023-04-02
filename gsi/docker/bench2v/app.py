import os
from logging import getLogger
from fastapi import FastAPI, Response, status
from vectorizer import Vectorizer, VectorInput
from meta import Meta
import dset

VERBOSE = False

app = FastAPI()
vec : Vectorizer
meta_config : Meta
logger = getLogger('uvicorn')

@app.on_event("startup")
def startup_event():
    dset.load()


@app.get("/.well-known/live", response_class=Response)
@app.get("/.well-known/ready", response_class=Response)
def live_and_ready(response: Response):
    response.status_code = status.HTTP_204_NO_CONTENT


@app.get("/meta")
def meta():
    return { 'model': "bench2v" }



@app.post("/vectors")
@app.post("/vectors/")
async def read_item(item: VectorInput, response: Response):
    try:
        #GW vector = await vec.vectorize(item.text, item.config)
        items = item.text.split()
        if VERBOSE: print(item, item.text, items )
        if len(items)==1:
            idx = int(items[0].strip())
        else: 
            idx = int( items[-1].strip())
        if VERBOSE: print("idx=",idx)
        vector = dset.get(idx)
        return {"text": item.text, "vector": vector.tolist(), "dim": len(vector)}
    except Exception as e:
        logger.exception(
            'Something went wrong while vectorizing data.'
        )
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error": str(e)}
