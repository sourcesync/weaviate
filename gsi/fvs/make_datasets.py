#
# Standard imports
#
import os
import sys
import struct

#
# Installed/external packages
#
import numpy
import datasets

#
# Constants
#

# if dataset file is found, verify it look's ok,
# this could take a while for large numpy files
VERIFY=False

# Store/retrieve bigann competition datasets at/to this location
BIGANN_COMP_DATA = "/mnt/nas1/fvs_benchmark_datasets/bigann_competition_data/"

# Store/retrieve FVS benchmark datasets at/to this location
FVS_DATA_DIR = "/mnt/nas1/fvs_benchmark_datasets/"

# Deep1B query set filenames
DEEP1B_QUERY_OG_SET = "deep-queries.npy"
DEEP1B_QUERY_1000_SET = "deep-queries-1000.npy"
DEEP1B_QUERY_100_SET = "deep-queries-100.npy"
DEEP1B_QUERY_10_SET = "deep-queries-10.npy"

# Deep20M filenames
DEEP20M =  "deep-20M.npy"
DEEP20M_GT_1000 = "deep-20M-gt-1000.npy"
DEEP20M_GT_100 = "deep-20M-gt-100.npy"
DEEP20M_GT_10 = "deep-20M-gt-10.npy"

# Deep30M filenames
DEEP30M =  "deep-30M.npy"
DEEP30M_GT_1000 = "deep-30M-gt-1000.npy"
DEEP30M_GT_100 = "deep-30M-gt-100.npy"
DEEP30M_GT_10 = "deep-30M-gt-10.npy"

# Deep40M filenames
DEEP40M =  "deep-40M.npy"
DEEP40M_GT_1000 = "deep-40M-gt-1000.npy"
DEEP40M_GT_100 = "deep-40M-gt-100.npy"
DEEP40M_GT_10 = "deep-40M-gt-10.npy"

# Deep45M filenames
DEEP45M =  "deep-45M.npy"
DEEP45M_GT_1000 = "deep-45M-gt-1000.npy"
DEEP45M_GT_100 = "deep-45M-gt-100.npy"
DEEP45M_GT_10 = "deep-45M-gt-10.npy"

# Deep50M filenames
DEEP50M =  "deep-50M.npy"
DEEP50M_GT_1000 = "deep-50M-gt-1000.npy"
DEEP50M_GT_100 = "deep-50M-gt-100.npy"
DEEP50M_GT_10 = "deep-50M-gt-10.npy"

# Deep60M filenames
DEEP60M =  "deep-60M.npy"
DEEP60M_GT_1000 = "deep-60M-gt-1000.npy"
DEEP60M_GT_100 = "deep-60M-gt-100.npy"
DEEP60M_GT_10 = "deep-60M-gt-10.npy"

# Deep70M filenames
DEEP70M =  "deep-70M.npy"
DEEP70M_GT_1000 = "deep-70M-gt-1000.npy"
DEEP70M_GT_100 = "deep-70M-gt-100.npy"
DEEP70M_GT_10 = "deep-70M-gt-10.npy"

# Deep80M filenames
DEEP80M =  "deep-80M.npy"
DEEP80M_GT_1000 = "deep-80M-gt-1000.npy"
DEEP80M_GT_100 = "deep-80M-gt-100.npy"
DEEP80M_GT_10 = "deep-80M-gt-10.npy"

# Deep90M filenames
DEEP90M =  "deep-90M.npy"
DEEP90M_GT_1000 = "deep-90M-gt-1000.npy"
DEEP90M_GT_100 = "deep-90M-gt-100.npy"
DEEP90M_GT_10 = "deep-90M-gt-10.npy"

# Deep250M filenames
DEEP250M =  "deep-250M.npy"
DEEP250M_GT_1000 = "deep-250M-gt-1000.npy"
DEEP250M_GT_100 = "deep-250M-gt-100.npy"
DEEP250M_GT_10 = "deep-250M-gt-10.npy"

# Deep1M filenames
DEEP1M =  "deep-1M.npy"
DEEP1M_GT_10 = "deep-1M-gt-10.npy"
DEEP1M_GT_10_DISTS = "deep-1M-gt-10-dists.npy"

# Deep10K filenames
DEEP10K =  "deep-10K.npy"
DEEP10K_GT_1000 = "deep-10K-gt-1000.npy"
DEEP10K_GT_10 = "deep-10K-gt-10.npy"

# Deep150M filenames
DEEP150M =  "deep-150M.npy"
DEEP150M_GT_1000 = "deep-150M-gt-1000.npy"
DEEP150M_GT_100 = "deep-150M-gt-100.npy"
DEEP150M_GT_10 = "deep-150M-gt-10.npy"


# 
# Configure modules
#

# Override the relative default "data" dir and point to NAS storage
datasets.BASEDIR = BIGANN_COMP_DATA

#
# Functions
#
def append_floatarray(fname, arr):
    '''This will create/append to a numpy file and add vectors to it.'''

# Deep60M filenames
DEEP60M =  "deep-60M.npy"

# Deep1M filenames
DEEP1M =  "deep-1M.npy"
DEEP1M_GT_10 = "deep-1M-gt-10.npy"
DEEP1M_GT_10_DISTS = "deep-1M-gt-10-dists.npy"

# Deep10K filenames
DEEP10K =  "deep-10K.npy"
DEEP10K_GT_1000 = "deep-10K-gt-1000.npy"
DEEP10K_GT_10 = "deep-10K-gt-10.npy"


# 
# Configure modules
#

# Override the relative default "data" dir and point to NAS storage
datasets.BASEDIR = BIGANN_COMP_DATA

#
# Functions
#
def append_floatarray(fname, arr):
    '''This will create/append to a numpy file and add vectors to it.'''

    if len(arr.shape)!=2:
        raise Exception("expected an ndarray of two dimenions") 

# Deep1M filenames
DEEP1M =  "deep-1M.npy"
DEEP1M_GT_10 = "deep-1M-gt-10.npy"
DEEP1M_GT_10_DISTS = "deep-1M-gt-10-dists.npy"

# Deep10K filenames
DEEP10K =  "deep-10K.npy"
DEEP10K_GT_1000 = "deep-10K-gt-1000.npy"
DEEP10K_GT_10 = "deep-10K-gt-10.npy"


# 
# Configure modules
#

# Override the relative default "data" dir and point to NAS storage
datasets.BASEDIR = BIGANN_COMP_DATA

#
# Functions
#
def append_floatarray(fname, arr):
    '''This will create/append to a numpy file and add vectors to it.'''

    if len(arr.shape)!=2:
        raise Exception("expected an ndarray of two dimenions") 

    # declare the special bytes for the numpy header
    preheader = b'\x93\x4e\x55\x4d\x50\x59\x01\x00\x76\x00'
    fmt_header = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }"
    empty = b'\x20'
    fin = b'\x0a'

    # Get file descriptor and determine create/append mode
    # as well as current size if in append mode.
    append = False
    cur_items = 0
    fsize = 0
    f = None
    if os.path.exists(fname):
        fsize = os.path.getsize(fname)
        append = True
        if (fsize-128) % (arr.shape[1]*4) != 0:
            raise Exception("unexpected file size (%d,%d,%d)" % ( fsize, fsize-128, arr.shape[1] ) )
        cur_items = int( (fsize-128) / (arr.shape[1]*4) )
        f = open(fname,"r+b")
    else:
        f = open(fname,"wb")
        append = False
 
    # 
    # Write numpy header
    #
    f.seek(0)
    idx =0
    for i in range(len(preheader)):
        f.write( bytes([preheader[i]]) )
        idx += 1
    header = bytes( fmt_header % (cur_items+arr.shape[0],arr.shape[1]), 'ascii' )
    for i in range(len(header)):
        f.write( bytes([header[i]]) )
        idx += 1
    for i in range(idx, 127):
        f.write( bytes([empty[0]]) )
        idx += 1
    f.write( bytes([fin[0]]) )

    #
    # Append the array to the end of the file
    #
    if append:
        f.seek( fsize )
    for i in range(arr.shape[0]):
        flist = arr[i].tolist()
        buf = struct.pack( '%sf' % len(flist), *flist)
        f.write(buf)
    f.flush()
    f.close()

    return (cur_items+arr.shape[0],arr.shape[1])

def test_append():
    '''A unit test for the append function above.'''
    arr = numpy.ones( (2, 96) )
    append_floatarray("./test.npy", arr)
    arr = numpy.load("./test.npy")
    print("test=", arr)

def make_deep1B_queries_subsets():

    if not os.path.exists( os.path.join(FVS_DATA_DIR, DEEP1B_QUERY_OG_SET) ):
        raise Exception("OG Deep1B query set does not exists->%s" %  DEEP1B_QUERY_OG_SET)

    # Get the full queries set
    ds = datasets.DATASETS["deep-1M"]() # Note that we could supply any subseet of Deep-1B here
    ds.prepare(False)
    queries = ds.get_queries()

    # Make q=1000 subsets
    subdir = os.path.join(FVS_DATA_DIR, "deep-1B-q1000-subsets")
    if not os.path.exists( subdir ):
        print("Making subdir", subdir)
        os.makedirs( subdir, exist_ok=False)
    num_subsets = int( queries.shape[0] / 1000 ) # floor
    for i in range(num_subsets):
        fname = "deep-queries-%d-%d.npy" % (i*1000+1, i*1000+1000)
        fpath = os.path.join( subdir, fname )
        if not os.path.exists(fpath):
            q = queries[i*1000:i*1000+1000,:]
            print("saving",fpath)
            numpy.save( fpath, q )   
        elif VERIFY:
            # Verify it
            print("Found %s.  Verifying it (this may take a sec.)" % fpath)
            arr = numpy.load(fpath)
            if arr.shape[0]!=1000:
                raise Exception("Invalid size.")
        
    # Make q=100 subsets
    subdir = os.path.join(FVS_DATA_DIR, "deep-1B-q100-subsets")
    if not os.path.exists( subdir ):
        print("Making subdir", subdir)
        os.makedirs( subdir, exist_ok=False)
    for i in range(10): # for now, we are only doing 10
        fname = "deep-queries-%d-%d.npy" % (i*100+1, i*100+100)
        fpath = os.path.join( subdir, fname )
        if not os.path.exists(fpath):
            q = queries[i*100:i*100+100,:]
            print("saving",fpath)
            numpy.save( fpath, q )
        elif VERIFY:
            # Verify it
            print("Found %s.  Verifying it (this may take a sec.)" % fpath)
            arr = numpy.load(fpath)
            if arr.shape[0]!=100:
                raise Exception("Invalid size.")
 
    # Make q=10 subsets
    subdir = os.path.join(FVS_DATA_DIR, "deep-1B-q10-subsets")
    if not os.path.exists( subdir ):
        print("Making subdir", subdir)
        os.makedirs( subdir, exist_ok=False)
    for i in range(100): # for now, we are only doing 100
        fname = "deep-queries-%d-%d.npy" % (i*10+1, i*10+10)
        fpath = os.path.join( subdir, fname )
        if not os.path.exists(fpath):
            q = queries[i*10:i*10+10,:]
            print("saving",fpath)
            numpy.save( fpath, q )
        elif VERIFY:
            # Verify it
            print("Found %s.  Verifying it (this may take a sec.)" % fpath)
            arr = numpy.load(fpath)
            if arr.shape[0]!=10:
                raise Exception("Invalid size.")

    # Make q=1 subsets
    subdir = os.path.join(FVS_DATA_DIR, "deep-1B-q1-subsets")
    if not os.path.exists( subdir ):
        print("Making subdir", subdir)
        os.makedirs( subdir, exist_ok=False)
    for i in range(1000): # for now, we are only doing 1000
        fname = "deep-queries-%d-%d.npy" % (i+1, i+1)
        fpath = os.path.join( subdir, fname )
        if not os.path.exists(fpath):
            q = queries[i:i+1,:]
            print("saving",fpath)
            numpy.save( fpath, q )
        elif VERIFY:
            # Verify it
            print("Found %s.  Verifying it (this may take a sec.)" % fpath)
            arr = numpy.load(fpath)
            if arr.shape[0]!=1:
                raise Exception("Invalid size.")


# Verify Deep1B original queries
fpath = os.path.join(FVS_DATA_DIR, DEEP1B_QUERY_OG_SET)
print("Checking ", fpath,"exists...")
if not os.path.exists(fpath):
    raise Exception("Deep1B original queries dataset does not exist->%s" %fpath)
elif VERIFY:
    # TODO
    pass 

# Make all the Deep1B queries subsets
make_deep1B_queries_subsets()


# Create/verify deep-20M
fname = os.path.join( FVS_DATA_DIR, DEEP20M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")

    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-20M"]()
    ds.prepare(True)

    for dt in ds.get_dataset_iterator(bs=1000):
        newsize = append_floatarray(fname, dt)
        print("Making deep-20M, appended batch, newsize=", newsize)
        if newsize[0]==20000000:
            break

    print("done") 

    if False:
        print("counting...")
        count = 0
        for dt in ds.get_dataset_iterator():
            count += 1
        print("%d" % count, type(dt), dt.shape, dt.dtype)

        arr = numpy.empty( (0,96), dt.dtype )
        print("arr shape", dt.shape)

        print("appending")
        for dt in ds.get_dataset_iterator():
            arr = numpy.concatenate( (arr, dt), axis=0 )
            print(dt.shape, arr.shape)

        print("saving",fname)
        numpy.save( fname, arr )
        print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=20000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-30M
fname = os.path.join( FVS_DATA_DIR, DEEP30M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")

    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-30M"]()
    ds.prepare(False)

    for dt in ds.get_dataset_iterator(bs=1000):
        newsize = append_floatarray(fname, dt)
        print("Making deep-30M, appended batch, newsize=", newsize)
        if newsize[0]==30000000:
            break

    print("done")

    if False:
        print("counting...")
        count = 0
        for dt in ds.get_dataset_iterator():
            count += 1
        print("%d" % count, type(dt), dt.shape, dt.dtype)

        arr = numpy.empty( (0,96), dt.dtype )
        print("arr shape", dt.shape)

        print("appending")
        for dt in ds.get_dataset_iterator():
            arr = numpy.concatenate( (arr, dt), axis=0 )
            print(dt.shape, arr.shape)

        print("saving",fname)
        numpy.save( fname, arr )
        print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=30000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


# DEEP30M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP30M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-30M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP30M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP30M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-30M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP30M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP30M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-30M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")



# Create/verify deep-40M
fname = os.path.join( FVS_DATA_DIR, DEEP40M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")

    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-40M"]()
    ds.prepare(False)

    for dt in ds.get_dataset_iterator(bs=1000):
        newsize = append_floatarray(fname, dt)
        print("Making deep-40M, appended batch, newsize=", newsize)
        if newsize[0]==40000000:
            break

    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=40000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


# DEEP40M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP40M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-40M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP40M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP40M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-40M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP40M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP40M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-40M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-45M
fname = os.path.join( FVS_DATA_DIR, DEEP45M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")

    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-45M"]()
    ds.prepare(False)

    for dt in ds.get_dataset_iterator(bs=1000):
        newsize = append_floatarray(fname, dt)
        print("Making deep-45M, appended batch, newsize=", newsize)
        if newsize[0]==45000000:
            break

    print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=45000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP45M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP45M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-45M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP45M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP45M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-45M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP45M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP45M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-45M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


#
# DEEP1B query set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP1B_QUERY_1000_SET )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-20M"]()
    ds.prepare(False)

    queries = ds.get_queries()
    print(queries.shape)
    queries = queries[:1000,:]
    print(queries.shape)

    print("saving",fname)
    numpy.save( fname, queries )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


# DEEP20M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP20M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-20M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


# DEEP1B query set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP1B_QUERY_100_SET )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-20M"]()
    ds.prepare(False)

    queries = ds.get_queries()
    print(queries.shape)
    queries = queries[:100,:]
    print(queries.shape)

    print("saving",fname)
    numpy.save( fname, queries )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


# DEEP20M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP20M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-20M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP1B query set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP1B_QUERY_10_SET )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-20M"]()
    ds.prepare(False)

    queries = ds.get_queries()
    print(queries.shape)
    queries = queries[:10,:]
    print(queries.shape)

    print("saving",fname)
    numpy.save( fname, queries )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


# DEEP20M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP20M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-20M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-50M
fname = os.path.join( FVS_DATA_DIR, DEEP50M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")

    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-50M"]()
    ds.prepare(True)

    for dt in ds.get_dataset_iterator(bs=1000):
        newsize = append_floatarray(fname, dt)
        print("deep-50M, appended batch, newsize=", newsize)
        if newsize[0]==50000000:
            break

    print("done") 

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=50000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP50M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP50M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-50M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


# DEEP50M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP50M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-50M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


# DEEP50M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP50M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-50M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-1M
fname = os.path.join( FVS_DATA_DIR, DEEP1M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")

    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-1M"]()
    ds.prepare(True)

    for dt in ds.get_dataset_iterator(bs=1000):
        newsize = append_floatarray(fname, dt)
        print("deep-1M, appended batch, newsize=", newsize)
        if newsize[0]==1000000:
            break

    print("done")

    if False:
        print("counting...")
        count = 0
        for dt in ds.get_dataset_iterator():
            count += 1
        print("%d" % count, type(dt), dt.shape, dt.dtype)

        arr = numpy.empty( (0,96), dt.dtype )
        print("arr shape", dt.shape)

        print("appending")
        for dt in ds.get_dataset_iterator():
            arr = numpy.concatenate( (arr, dt), axis=0 )
            print(dt.shape, arr.shape)

        print("saving",fname)
        numpy.save( fname, arr )
        print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP1M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP1M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-1M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP1M of DEEP1B, gt set - 10 - dists
fname = os.path.join( FVS_DATA_DIR, DEEP1M_GT_10_DISTS )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-1M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    D = D[:10,:]
    print(D.shape)

    print("saving",fname)
    numpy.save( fname, D )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-10K
fname = os.path.join( FVS_DATA_DIR, DEEP10K )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")

    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-10K"]()
    ds.prepare(False)

    for dt in ds.get_dataset_iterator(bs=1000):
        newsize = append_floatarray(fname, dt)
        print("deep-10K, appended batch, newsize=", newsize)
        if newsize[0]==10000:
            break

    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP10K of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP10K_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-10K"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP10K of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP10K_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-10K"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-60M
fname = os.path.join( FVS_DATA_DIR, DEEP60M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")

    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-60M"]()
    ds.prepare(skip_data=False, skip_non_data=True)

    for dt in ds.get_dataset_iterator(bs=10000):
        newsize = append_floatarray(fname, dt)
        print("deep-60M, appended batch, newsize=", newsize)
        if newsize[0]==60000000:
            break

    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=60000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP60M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP60M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-60M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP60M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP60M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-60M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP60M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP60M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-60M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-70M
fname = os.path.join( FVS_DATA_DIR, DEEP70M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")
    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-70M"]()
    ds.prepare(skip_data=False, skip_non_data=True)

    for dt in ds.get_dataset_iterator(bs=10000):
        newsize = append_floatarray(fname, dt)
        print("deep-70M, appended batch, newsize=", newsize)
        if newsize[0]==70000000:
            break
    print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=70000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP70M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP70M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-70M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP70M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP70M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-70M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP70M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP70M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-70M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-80M
fname = os.path.join( FVS_DATA_DIR, DEEP80M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")
    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-80M"]()
    ds.prepare(skip_data=False, skip_non_data=True)

    for dt in ds.get_dataset_iterator(bs=10000):
        newsize = append_floatarray(fname, dt)
        print("deep-80M, appended batch, newsize=", newsize)
        if newsize[0]==80000000:
            break
    print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=80000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP80M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP80M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-80M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP80M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP80M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-80M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP80M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP80M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-80M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-90M
fname = os.path.join( FVS_DATA_DIR, DEEP90M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")
    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-90M"]()
    ds.prepare(skip_data=False, skip_non_data=True)

    for dt in ds.get_dataset_iterator(bs=10000):
        newsize = append_floatarray(fname, dt)
        print("deep-90M, appended batch, newsize=", newsize)
        if newsize[0]==90000000:
            break
    print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=90000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP90M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP90M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-90M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP90M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP90M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-90M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP90M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP90M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-90M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


# Create/verify deep-250M
fname = os.path.join( FVS_DATA_DIR, DEEP250M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")
    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-250M"]()
    ds.prepare(skip_data=False, skip_non_data=True)

    for dt in ds.get_dataset_iterator(bs=10000):
        newsize = append_floatarray(fname, dt)
        print("deep-250M, appended batch, newsize=", newsize)
        if newsize[0]==250000000:
            break
    print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=250000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP250M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP250M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-250M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP250M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP250M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-250M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP250M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP250M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-250M"]()
    ds.prepare(False)

    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)

    print("saving",fname)
    numpy.save( fname, I )
    print("done")

elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# Create/verify deep-150M
fname = os.path.join( FVS_DATA_DIR, DEEP150M )
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    print("Creating", fname, "...")
    print("Downloading Competition Deep1B base, query, and gt...")
    ds = datasets.DATASETS["deep-150M"]()
    ds.prepare(skip_data=False, skip_non_data=True)

    for dt in ds.get_dataset_iterator(bs=10000):
        newsize = append_floatarray(fname, dt)
        print("deep-250M, appended batch, newsize=", newsize)
        if newsize[0]==150000000:
            break
    print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it (this may take a sec.)" % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=150000000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP150M of DEEP1B, gt set - 1000
fname = os.path.join( FVS_DATA_DIR, DEEP150M_GT_1000)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-150M"]()
    ds.prepare(False)
    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:1000,:]
    print(I.shape)
    print("saving",fname)
    numpy.save( fname, I )
    print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=1000:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP150M of DEEP1B, gt set - 100
fname = os.path.join( FVS_DATA_DIR, DEEP150M_GT_100)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-150M"]()
    ds.prepare(False)
    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:100,:]
    print(I.shape)
    print("saving",fname)
    numpy.save( fname, I )
    print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=100:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")

# DEEP150M of DEEP1B, gt set - 10
fname = os.path.join( FVS_DATA_DIR, DEEP150M_GT_10)
print("Checking ", fname,"exists...")
if not os.path.exists(fname):
    ds = datasets.DATASETS["deep-150M"]()
    ds.prepare(False)
    I, D = ds.get_groundtruth()
    print(I.shape)
    I = I[:10,:]
    print(I.shape)
    print("saving",fname)
    numpy.save( fname, I )
    print("done")
elif VERIFY:
    # Verify it
    print("Found %s.  Verifying it..." % fname)
    arr = numpy.load(fname)
    if arr.shape[0]!=10:
        raise Exception("Bad size for %s" % fname, arr.shape)
    print("Verified.")


print("Done.")
