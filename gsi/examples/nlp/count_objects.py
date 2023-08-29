import weaviate
client = weaviate.Client('http://localhost:8091')

try:
    count = client.query.aggregate('News').with_meta_count().do()
    print(count)
except Exception:
    raise Exception
