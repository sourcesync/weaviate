#
# Imports
#

import argparse
import logging
import time
import resource
import pdb
import os

import numpy
import numpy as np
import faiss
from faiss.contrib.exhaustive_search import range_search_gpu

#
# Config
#

# Default to KNN style search
SEARCH_TYPE = "knn"

#
# Globals
#

# Will contain the total records of the base dataset
TOTAL_SIZE = 0

class ResultHeap:
    """Accumulate query results from a sliced dataset. The final result will
    be in self.D, self.I."""

    def __init__(self, nq, k, keep_max=False):
        " nq: number of query vectors, k: number of results per query "
        self.I = np.zeros((nq, k), dtype='int64')
        self.D = np.zeros((nq, k), dtype='float32')
        self.nq, self.k = nq, k
        if keep_max:
            heaps = faiss.float_minheap_array_t()
        else:
            heaps = faiss.float_maxheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(self.D)
        heaps.ids = faiss.swig_ptr(self.I)
        heaps.heapify()
        self.heaps = heaps

    def add_result(self, D, I):
        """D, I do not need to be in a particular order (heap or sorted)"""
        assert D.shape == (self.nq, self.k)
        assert I.shape == (self.nq, self.k)
        self.heaps.addn_with_ids(
            self.k, faiss.swig_ptr(D),
            faiss.swig_ptr(I), self.k)

    def finalize(self):
        self.heaps.reorder()

def sanitize(x):
    return numpy.ascontiguousarray(x, dtype='float32')

def get_dataset_iterator(dfile, bs=512, split=(1,0)):
    print("Getting dataset iterator for", dfile)

    #TODO: you should consider an memory map version for large datasets
    #x = xbin_mmap(filename, dtype=self.dtype, maxn=self.nb)
    x = np.load(dfile)
    print("Dataset shape", x.shape)
    global TOTAL_SIZE
    TOTAL_SIZE = x.shape[0]

    # We generate batch size chunks of the base dataset
    # and return as a python generator to avoid using too 
    # much main memory.
    nb = x.shape[0]
    nsplit, rank = split
    i0, i1 = nb * rank // nsplit, nb * (rank + 1) // nsplit
    #assert x.shape == (self.nb, self.d)
    for j0 in range(i0, i1, bs):
        j1 = min(j0 + bs, i1)
        yield sanitize(x[j0:j1])

def knn_ground_truth(dfile, qfile, k, bs, split):
    """Computes the exact KNN search results for a dataset that possibly
    does not fit in RAM but for which we have an iterator that
    returns it block by block.
    """
    print("loading queries...")
    xq = np.load( qfile )
    print("knn_ground_truth: queries shape", xq.shape)

    print("normalizing query...")
    xq = xq / np.linalg.norm(xq)
    print("normalizing query shape: ", xq.shape)
    print("checking query normalization... ")
    if np.linalg.norm(xq) > 0.9:
        print("query: ", np.linalg.norm(xq))
    else:
        print("query < 0.9: ", np.linalg.norm(xq))
        exit()

    #if ds.distance() == "angular":
    #    faiss.normalize_L2(xq)

    print("knn_ground_truth queries size %s k=%d" % (xq.shape, k))

    t0 = time.time()
    nq, d = xq.shape

    ##metric_type = (
    #    faiss.METRIC_L2 if ds.distance() == "euclidean" else
    #    faiss.METRIC_INNER_PRODUCT if ds.distance() in ("ip", "angular") else
    #    1/0
    #)
    metric_type = faiss.METRIC_L2
    rh = ResultHeap(nq, k, keep_max=metric_type == faiss.METRIC_INNER_PRODUCT)

    index = faiss.IndexFlat(d, metric_type)

    if faiss.get_num_gpus():
        print('running on %d GPUs' % faiss.get_num_gpus())
        index = faiss.index_cpu_to_all_gpus(index)

    # compute ground-truth by chunks, and add to heaps
    i0 = 0
    for xbi in get_dataset_iterator(dfile, bs=bs, split=split):
        ni = xbi.shape[0]

        print("normalizing base...")
        xbi = xbi / np.linalg.norm(xbi)
        print("normalizing base shape: ", xbi.shape)
        print("checking base normalization... ")
        if np.linalg.norm(xbi) > 0.9:
            print("base: ", np.linalg.norm(xbi))
        else:
            print("base < 0.9: ", np.linalg.norm(xbi))
            exit()


        ##if ds.distance() == "angular":
        ##    faiss.normalize_L2(xbi)
        index.add(xbi)
        D, I = index.search(xq, k)
        I += i0
        rh.add_result(D, I)
        index.reset()
        i0 += ni
        print(f"[{time.time() - t0:.2f} s] {i0} / {TOTAL_SIZE} vectors", end="\r", flush=True)

    rh.finalize()
    print()
    print("GT time: %.3f s (%d vectors)" % (time.time() - t0, i0))

    return rh.D, rh.I


def range_ground_truth(ds, radius, bs, split):
    """Computes the exact range search results for a dataset that possibly
    does not fit in RAM but for which we have an iterator that
    returns it block by block.
    """
    print("loading queries")
    xq = ds.get_queries()

    if ds.distance() == "angular":
        faiss.normalize_L2(xq)

    print("range_ground_truth queries size %s radius=%g" % (xq.shape, radius))

    t0 = time.time()
    nq, d = xq.shape

    metric_type = (
        faiss.METRIC_L2 if ds.distance() == "euclidean" else
        faiss.METRIC_INNER_PRODUCT if ds.distance() in ("ip", "angular") else
        1/0
    )

    index = faiss.IndexFlat(d, metric_type)

    if faiss.get_num_gpus():
        print('running on %d GPUs' % faiss.get_num_gpus())
        index_gpu = faiss.index_cpu_to_all_gpus(index)
    else:
        index_gpu = None

    results = []

    # compute ground-truth by blocks, and add to heaps
    i0 = 0
    tot_res = 0
    for xbi in ds.get_dataset_iterator(bs=bs, split=split):
        ni = xbi.shape[0]
        if ds.distance() == "angular":
            faiss.normalize_L2(xbi)

        index.add(xbi)
        if index_gpu is None:
            lims, D, I = index.range_search(xq, radius)
        else:
            index_gpu.add(xbi)
            lims, D, I = range_search_gpu(xq, radius, index_gpu, index)
            index_gpu.reset()
        index.reset()
        I = I.astype("int32")
        I += i0
        results.append((lims, D, I))
        i0 += ni
        tot_res += len(D)
        print(f"[{time.time() - t0:.2f} s] {i0} / {ds.nb} vectors, {tot_res} matches",
            end="\r", flush=True)
    print()
    print("merge into single table")
    # merge all results in a single table
    nres = np.zeros(nq, dtype="int32")
    D = []
    I = []
    for q in range(nq):
        nres_q = 0
        for lims_i, Di, Ii in results:
            l0, l1 = lims_i[q], lims_i[q + 1]
            if l1 > l0:
                nres_q += l1 - l0
                D.append(Di[l0:l1])
                I.append(Ii[l0:l1])
        nres[q] = nres_q

    D = np.hstack(D)
    I = np.hstack(I)
    assert len(D) == nres.sum() == len(I)
    print("GT time: %.3f s (%d vectors)" % (time.time() - t0, i0))
    return nres, D, I

def usbin_write(ids, dist, fname):
    ids = np.ascontiguousarray(ids, dtype="int32")
    dist = np.ascontiguousarray(dist, dtype="float32")
    assert ids.shape == dist.shape
    f = open(fname, "wb")
    n, d = dist.shape
    np.array([n, d], dtype='uint32').tofile(f)
    ids.tofile(f)
    dist.tofile(f)


def range_result_write(nres, I, D, fname):
    """ write the range search file format:
    int32 n_queries
    int32 total_res
    int32[n_queries] nb_results_per_query
    int32[total_res] database_ids
    float32[total_res] distances
    """
    nres = np.ascontiguousarray(nres, dtype="int32")
    I = np.ascontiguousarray(I, dtype="int32")
    D = np.ascontiguousarray(D, dtype="float32")
    assert I.shape == D.shape
    total_res = nres.sum()
    nq = len(nres)
    assert I.shape == (total_res, )
    f = open(fname, "wb")
    np.array([nq, total_res], dtype='uint32').tofile(f)
    nres.tofile(f)
    I.tofile(f)
    D.tofile(f)


if __name__ == "__main__":

    #
    # parse cmd line args
    #
    parser = argparse.ArgumentParser()
    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('dataset options')
    aa('--dataset', required=True)
    aa('--queries', required=True)
    aa('--split', type=int, nargs=2, default=[1, 0],
        help="split that must be handled")
    group = parser.add_argument_group('computation options')
    aa('--range_search', action="store_true", help="do range search instead of kNN search")
    aa('--k', default=100, type=int, help="number of nearest kNN neighbors to search")
    aa('--radius', default=96237, type=float, help="range search radius")
    aa('--bs', default=100_000, type=int, help="batch size for database iterator")
    aa("--maxRAM", default=100, type=int, help="set max RSS in GB (avoid OOM crash)")
    group = parser.add_argument_group('output options')
    aa('--o', default="", help="output file name")
    aa('--numpy',default=False,help="use numpy format")
    args = parser.parse_args()

    #
    # check args
    #
    if args.maxRAM > 0:
        print("setting max RSS to", args.maxRAM, "GiB")
        resource.setrlimit(
            resource.RLIMIT_DATA, (args.maxRAM * 1024 ** 3, resource.RLIM_INFINITY)
        )
    ds = args.dataset
    if not os.path.exists(ds):
        raise Exception("Path to base dataset %s does not exists", ds)
    qu = args.queries
    if not os.path.exists(qu):
        raise Exception("Path to queries dataset %s does not exists", qu)

    #
    # compute reults based on search type
    #
    if not args.range_search: # traditional knn search 
        D, I = knn_ground_truth(ds, qu, k=args.k, bs=args.bs, split=args.split)
        print(f"Writing index matrix of size {I.shape} to {args.o}")
        # write in the usbin format
        if args.numpy:
            print("Using numpy format for", I.shape)
            np.save(args.o,I)
            print("Saved file at", args.o)
        else:
            usbin_write(I, D, args.o)
    else:
        nres, D, I = range_ground_truth(ds, radius=args.radius, bs=args.bs, split=args.split)
        print(f"Writing results {I.shape} to {args.o}")
        range_result_write(nres, I, D, args.o)

