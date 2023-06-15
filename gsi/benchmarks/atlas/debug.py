import numpy as np
import sys
import os
from sklearn.neighbors import NearestNeighbors
import faiss

NAS="/mnt/nas1/atlas_data/benchmarking/"
ORIG_BASE="base_atlas.npy"
ORIG_QUERY="query_vec.npy"
NORM_SUBSET="base_atlas_norm_%d.npy"
NORM_QUERY="query_vec_norm.npy"
NORM_GT="gt_from_norm_%d.npy"
SUBSET_SZ=50000

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
print("cosine", D.shape,I.shape, D[0],I[0])

# reverse for cosine debugging
#Dr = np.flip(D)
#Ir = np.flip(I)
#print("cosine reverse", Ir)

# export ground truth array
fpath = os.path.join( NAS, NORM_GT % SUBSET_SZ )
np.save(fpath, I )
print("saved gt from norms",fpath)

# get GT via brute force with faiss
print("Getting brute force gt with faiss")
Dref, Iref = faiss.knn(query_norm, subset_norm, 10)
print("faiss", Iref[0])

# try a faiss index
print("Trying faiss L2 index...")
index = faiss.IndexFlatL2(subset_norm.shape[1])
index.add( subset_norm )
print(index.ntotal)
D, I = index.search(query_norm, 10) 
print("faiss index", I[0])

#
# try FAISS ANN index using normalized version of subset of atlas base
#
print("Trying faiss ANN index...")
ds = np.load( "/mnt/nas1/atlas_data/benchmarking/base_atlas_norm_%d.npy" % SUBSET_SZ)
nlist = 50 #clusters
quantizer = faiss.IndexFlatL2(ds.shape[1])
index = faiss.IndexIVFFlat(quantizer, ds.shape[1], nlist)
index.train( ds )
index.add( ds )
index.nprobe = 10
qs = np.load( "/mnt/nas1/atlas_data/benchmarking/query_vec_norm.npy")
D, I = index.search(query_norm, 10) 
print("faiss search result first query", I[0])
gt = np.load( "/mnt/nas1/atlas_data/benchmarking/gt_from_norm_%d.npy" % SUBSET_SZ)
print("ground truth first query", gt[0])
# compute the intersection of query results and ground truth of first query
intersection = np.intersect1d( I[0][0:10], gt[0][0:10] )
print("intersection", intersection)
print("recall", len(intersection)/10.0)




