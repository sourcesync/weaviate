import numpy as np
import sys
import os
from sklearn.neighbors import NearestNeighbors


NAS="/mnt/nas1/atlas_data/benchmarking/"
ORIG_BASE="base_atlas.npy"
ORIG_QUERY="query_vec.npy"
NORM_SUBSET="base_atlas_norm_%d.npy"
NORM_QUERY="query_vec_norm.npy"
NORM_GT="gt_from_norm_cosine_reverse.npy"
SUBSET_SZ=10000

# load the original atlas base embeddings
atlas = np.load( os.path.join(NAS,ORIG_BASE), mmap_mode="r")
print("atlas base original", atlas.shape)

# get 1st array
first = atlas[0]
mag = np.linalg.norm(first) 
print("first el", first.shape, "mag", mag)
if mag != 1.0:
    print("not normalized")

# extract a subset
subset = atlas[0:SUBSET_SZ,:]
print("subset", subset.shape, subset.dtype)

# calc norm of first el and verify it
first = subset[0]
mag =  np.linalg.norm(first)
print("first el mag", mag)
first_norm = first/mag
mag =  np.linalg.norm(first_norm)
print("first el mag", mag)

# perform normalization for entire subset
subset_norm = np.empty( subset.shape, subset.dtype )
print("subset norm shape,", subset_norm.shape)
for i in range( subset_norm.shape[0] ):
    subset_norm[i] = subset[i]/np.linalg.norm(subset[i])

# verify normalization for the new subset array
for i in range( subset_norm.shape[0] ):
    mag = np.linalg.norm( subset_norm[i] )
    if abs(mag-1.0)>0.001:
        print("Normalization check failed at %d" % i)
        sys.exit(1)
print("normalization check passed for subset", subset_norm.shape)

# export normalized subset array
fpath = os.path.join( NAS, NORM_SUBSET % SUBSET_SZ )
print("writing normalized base subset",fpath)
np.save(fpath, subset_norm )
print("saved normalized base subset",fpath)

# load and normalize query set
print("normaling query set...")
query = np.load( os.path.join(NAS, ORIG_QUERY) )
query_norm = np.empty( query.shape, query.dtype )
for i in range( query_norm.shape[0] ):
    query_norm[i] = query[i]/np.linalg.norm(query[i])
for i in range( query_norm.shape[0] ):
    mag = np.linalg.norm( query_norm[i] )
    if abs(mag-1.0)>0.001:
        print("Normalization check failed at %d" % i)
        sys.exit(1)
print("normalization check passed for queries", query_norm.shape)

# export normalized query array
fpath = os.path.join( NAS, NORM_QUERY )
print("writing normalized queries",fpath)
np.save(fpath, subset_norm )
print("saved normalized queries",fpath)

# create ground truth
print("Computing ground truth.")
nbrs = NearestNeighbors(n_neighbors=100, metric="cosine", algorithm='brute').fit(subset_norm)
D, I = nbrs.kneighbors(query_norm)
print(D.shape,I.shape, D[0])

# reverse for cosine debugging
Dr = np.flip(D)
Ir = np.flip(I)

# export normalized query array
fpath = os.path.join( NAS, NORM_GT )
np.save(fpath, Ir )
print("saved gt from norms",fpath)







