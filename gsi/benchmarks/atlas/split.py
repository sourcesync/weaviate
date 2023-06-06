import numpy as np
import os
import random
from tqdm import tqdm

# CONFIG
ATLAS_EMBEDDINGS = "/mnt/nas1/atlas_data/benchmarking/atlas.npy"
QUERY_INDS = "/mnt/nas1/atlas_data/benchmarking/query_ind.npy"
QUERY_VECS = "/mnt/nas1/atlas_data/benchmarking/query_vec.npy"
BASE_ATLAS = "/mnt/nas1/atlas_data/benchmarking/base_atlas.npy"

SIZE = 1000

# load the final embeddings
print("loading atlas embeddings...")
arr = np.load(ATLAS_EMBEDDINGS, allow_pickle=True)
TOTAL_LENGTH = arr.shape[0]

# make a copy of the arr
arr_cp = arr 

# generate 1000 random index
random.seed(30) # same every time

print("generating index...")

LST_IND = []
for i in range(SIZE):
    idx = random.randint(0, TOTAL_LENGTH)
    LST_IND.append(idx)

print("saving index...")
# save 1000 query index
np.save(QUERY_INDS, LST_IND)

print("generating query vectors...")
arr_lst = []
for idx in LST_IND:
    q_arr = arr[idx]#.reshape(1, 768)
    arr_lst.append(q_arr)
    #print("deleting ", idx, " from base atlas...")
    #arr_new = np.delete(arr_cp, idx)

print("saving query vectors...")
# save 1000 query vectors
np.save(QUERY_VECS, arr_lst)

# create base atlas
arr_new = []

for i in tqdm(range(TOTAL_LENGTH)):
    if i not in LST_IND:
        #arr_new = np.take(arr_cp, i)
        arr_new.append(arr[i])

print("saving base embeddings now.. ")
np.save(BASE_ATLAS, arr_new)
