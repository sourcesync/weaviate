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

# generate 1000 random index
random.seed(30) # same every time

LST_IND = []
for i in 

querys = np.random.choice(arr, size=SIZE, replacement=False)
np.save(QUERY_VECS, querys, allow_pickle=True)

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

append_floatarray( FINAL_FILE, arr )