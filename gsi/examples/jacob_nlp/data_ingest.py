import weaviate, os, time
# get docs from each subdir
DATADIR = "/mnt/nas1/news20/"
def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))
paths = absoluteFilePaths(DATADIR)

# initialize client and class object
client = weaviate.Client('http://localhost:8080') 
class_obj = {
    "class": "News",
    "vectorizer": "text2vec-transformers",
    "properties": [
        {
            "name": "newsType",
            "dataType": ["text"]
        },
        {
            "name": "text",
            "dataType": ["text"]
        }
    ]
}
try:
    client.schema.create_class(class_obj) 
except weaviate.UnexpectedStatusCodeException as e:
    print('Class already exists')
    pass

t_start = time.process_time()
# add data to weaviate schema
client.batch.batch_size, count = 100, 0
for i, path in enumerate(paths):
    if i == 15000:
        break
    with open(path, errors='ignore') as file:
        data = file.read()
        data_obj = {
            "newsType": path.split('/')[-2], # gets the directory for each doc
            "text": data # doc text
            }
        client.batch.add_data_object(data_object=data_obj, class_name="News")
        file.close()
        count += 1
    if count == client.batch.batch_size:
        res = client.batch.create_objects() # push docs to weaviate
        count = 0
    if i % 1000 == 0: # show status every 1000 docs
        print(f'docs added: {i}, latest doc: {path}')

client.batch.create_objects() # push remaining docs
t_end = time.process_time()
print('time elapsed:', t_end - t_start)