import numpy as np
import os
import random

# CONFIG
ATLAS_EMBEDDINGS = "/mnt/nas1/atlas_data/benchmarking/atlas.npy"
QUERY_INDS = "/mnt/nas1/atlas_data/benchmarking/query_ind.npy"
QUERY_VECS = "/mnt/nas1/atlas_data/benchmarking/query_vec.npy"
BASE_ATLAS = "/mnt/nas1/atlas_data/benchmarking/base_atlas.npy"

SIZE = 1000

# load the final embeddings
arr = np.load(ATLAS_EMBEDDINGS, allow_pickle=True)
TOTAL_LENGTH = arr.shape[0]

# make a copy of the arr
arr_cp = arr 

# generate 1000 random index
random.seed(30) # same every time

LST_IND = []
for i in range(SIZE):
    idx = random.randint(0, TOTAL_LENGTH)
    LST_IND.append(idx)

# save 1000 query index
np.save(QUERY_INDS, LST_IND)

arr_lst = []
for idx in LST_IND:
    q_arr = arr[idx]#.reshape(1, 768)
    arr_lst.append(q_arr)
    print("deleting ", idx, " from base atlas...")
    arr_new = np.delete(arr_cp, idx)

# save 1000 query vectors
np.save(QUERY_VECS, arr_lst)
np.save(BASE_ATLAS, arr_new)
