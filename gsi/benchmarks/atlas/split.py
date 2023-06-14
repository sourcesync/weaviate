import numpy as np
import os
import random
from tqdm import tqdm

# CONFIG
ATLAS_EMBEDDINGS = "/mnt/nas1/atlas_data/benchmarking/atlas.npy"

QUERY_INDS = "/mnt/nas1/atlas_data/benchmarking/sets_nor/query_ind.npy"
QUERY_VECS = "/mnt/nas1/atlas_data/benchmarking/sets_nor/query_vec.npy"
BASE_ATLAS = "/mnt/nas1/atlas_data/benchmarking/sets_nor/base_atlas.npy"

SIZE = 1000

# load the final embeddings
print("loading atlas embeddings...")
arr = np.load(ATLAS_EMBEDDINGS, allow_pickle=True)
TOTAL_LENGTH = arr.shape[0]

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
    q_arr = q_arr / np.linalg.norm(q_arr)
    
    if np.linalg.norm(q_arr) >= 0.9:
        print("checking normalization: ", np.linalg.norm(q_arr))
    else:
        print("not 1: ", np.linalg.norm(q_arr))
        exit()

    arr_lst.append(q_arr)

print("saving query vectors...")
# save 1000 query vectors
np.save(QUERY_VECS, arr_lst)

# create base atlas
arr_new = []

for i in tqdm(range(TOTAL_LENGTH)):
    if i not in LST_IND:
        v = arr[i]
        v = v / np.linalg.norm(v)
    
        if np.linalg.norm(v) >= 0.9:
            print("checking normalization: ", np.linalg.norm(v))
        else:
            print("not 1: ", np.linalg.norm(v))
            exit()

        arr_new.append(v)

print("saving base embeddings now.. ")
np.save(BASE_ATLAS, arr_new)
