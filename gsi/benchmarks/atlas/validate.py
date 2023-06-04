
import numpy as np

print("loading query inds...")
q_inds = np.load("/mnt/nas1/atlas_data/benchmarking/query_ind.npy")
print("query inds shape", q_inds.shape)
if q_inds.shape[0]!=1000:
    raise Exception("Invalid q inds recs size")

print("loading query vecs...")
q_vecs = np.load("/mnt/nas1/atlas_data/benchmarking/query_vec.npy")
print("query vecs shape", q_vecs.shape)
if q_vecs.shape[1]!=768:
    raise Exception("Invalid q vecs dims")
if q_vecs.shape[0]!=1000:
    raise Exception("Invalid q vecs recs size")

print("Loading atlas...")
atlas = np.load("/mnt/nas1/atlas_data/benchmarking/atlas.npy")
print("atlas shape", atlas.shape)
if atlas.shape[1]!=768:
    raise Exception("Invalid atlas dims")

print("loading base...")
base = np.load("/mnt/nas1/atlas_data/benchmarking/base_atlas.npy")
print("base shape", base.shape)
if base.shape[1]!=768:
    raise Exception("Invalid base dims")
if base.shape[0]!= (atlas.shape[0]-1000):
    raise Exception("Invalid base rec size")

print("Checking vectors...")
for i in range(1000):
    print("Checking vector (%d/1000)..." % (i+1) )
    qvec = q_vecs[i]
    ind = q_inds[i]
    bvec = atlas[ind]
    if list( np.squeeze(qvec))!=list( np.squeeze(bvec)):
        raise Exception("Invalid vec comparison at %d" % i )

print("Validation passed.")
