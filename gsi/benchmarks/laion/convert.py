import torch
import os
import numpy as np
import pickle
import struct
import sys

##################################
#### Configuration Stars Here ####

# Debug mode
DEBUG = False

# Directory of converted files
NPY_DIR = "/mnt/nas1/laion400m/benchmarking/npy"

# Atlas directory of original files
LAION_DIR = "/mnt/nas1/laion400m/images/img_emb"

# If True, try loading an atlas file as pickled object if pytorch load fails
# else if False just skip this file
TRY_PICKLE_LOAD = False

# The final file to store all
if DEBUG:
    FINAL_FILE = "/mnt/nas1/laion400m/benchmarking/laion24M.npy.test"
else:
    FINAL_FILE = "/mnt/nas1/laion400m/benchmarking/laion24M.npy"

# Verification properties for the final file
if DEBUG:
    FINAL_SHAPE = ( 949500, 768 )
    FINAL_DTYPE = "float32"
    VERIFY_ARRAY_DATA = [ ( "img_emb_0000.npy", 0, 0 ) ]
else:
    FINAL_SHAPE = ( 24656862 , 768 )
    FINAL_DTYPE = "float32"
    VERIFY_ARRAY_DATA = [ ( "img_emb_0000.npy", 0, 0 ), ("img_emb_0025.npy", -1, -1)]

##################################
#### Useful Functions ###########

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

##################################
#### Program Starts Here ####

if not os.path.exists( FINAL_FILE ) or DEBUG: # The conditions for creating the final file from scratch

    if DEBUG: # In DEBUG mode remove previous final test file (if any)
        if os.path.exists( FINAL_FILE ):
            print("Warning: In DEBUG mode, so removing previous test file at", FINAL_FILE)
            os.remove( FINAL_FILE )

    print("Creating npy file at path", FINAL_FILE )

    # Iterate all atlas files
    for idx, filename in enumerate(os.listdir(LAION_DIR)):

        print("Processing %s (%d of %d)..." % (filename, idx+1, len(os.listdir(LAION_DIR))) )

        """# Deal with source file conversion to npy as needed
        npy_name = filename.replace('.', '')[:-2] + ".npy"
        npy_fp = os.path.join(NPY_DIR, npy_name)
        if os.path.exists(npy_fp):
            print("Already processed", filename)
        else: 
            laion_file = os.path.join(LAION_DIR, filename)
            print("Converting filename", filename)

            # We try to treat is as a tensor embedding file 
            # otherwise its something else like original document text
            # and we can likely ignore it.
            try:
                # Load tensor
                #print("Trying to load file as torch tensor...")
                #tensor = torch.load( atlas_file, map_location=torch.device('cpu'))
                #print("Done loading as tensor.")

                # Convert to np array
                npy = np.load(laion_file, allow_pickle=True)
                print("Done loading npy embedding...")
                #print("Done converting to npy data...")

                # Save to .npy
                npy_name = filename.replace('.', '')[:-2] + ".npy"
                npy_fp = os.path.join(NPY_DIR, npy_name)
                np.save(npy_fp, npy)
                print("Wrote file at", npy_fp)

            except:
                if TRY_PICKLE_LOAD:
                    print("Torch load failed.  Trying to load as a pickled object...")
                    with open(fp, 'rb') as f:
                        obj = f.read()
                        data = pickle.loads(obj, encoding='latin1')
                        print( "Got data->", type(data) )
                else:
                    print("Skipping this file since its likely not an embedding file", filename)
                    continue"""

        # At this point, we have verified the converted file exists
        # or we converted to npy or we skipped it.  We can continue
        # constructing the final npy file with all embeddings.
        npy_fp = os.path.join(LAION_DIR, filename)
        print("Loading converted file at", npy_fp)
        arr = np.load( npy_fp )
        #print("Transposing...")
        #arr = np.transpose(arr)
        shape = arr.shape
        print("Shape=",shape, "dtype=",arr.dtype)

        print("Writing/appending to", FINAL_FILE)
        append_floatarray( FINAL_FILE, arr )

        # Stop at first file if in DEBUG mode
        if DEBUG:
            print("Warning: Early break because we are in DEBUG mode...")
            break

# At this point, we should have a final file with all embeddings.
# Let's go ahead and do some lightweight verification.

if os.path.exists( FINAL_FILE ): 

    print("Found final file at", FINAL_FILE, ".  Verifying it...")

    final_embeddings = np.load( FINAL_FILE, allow_pickle=True )

    print("shape=", final_embeddings.shape, "type=", final_embeddings.dtype)

    # verify the final shape expected
    if final_embeddings.shape != FINAL_SHAPE:
        print("ERROR: shape validation failed.  Expected", FINAL_SHAPE)
   
    # verify the final dtype expected 
    if final_embeddings.dtype!= FINAL_DTYPE:
        print("ERROR: dtype validation failed.  Expected", FINAL_DTYPE)

    # spot check elements of the final array using the directives
    # from the VERIFY_ARRAY_DATA structure
    if VERIFY_ARRAY_DATA:

        for verify in VERIFY_ARRAY_DATA:
            print("Spot checking using directive", verify, "...")

            # get path to original embeddings file after npy conversion
            converted_file = os.path.join( NPY_DIR, verify[0] )
            # load it
            arr = np.load( converted_file )
            arr = np.transpose(arr) # remember to transpose!

            # get the spot check element source from original embeddings
            src_el = arr[ verify[1] ]

            # get the spot check element target from final array
            target_el = final_embeddings[ verify[2] ]

            # check equality
            def check_arr_equal(src,target):
                if src.shape!=target.shape:
                    print("Shape check failed")
                    return False
                for i in range(src.shape[0]):
                    #if DEBUG:
                    #   print("Comparing float values at position %d, %f == %f ?" % (i, src[i], target[i]))
                    if src[i]!=target[i]:
                        print("Float equality check failed at", i, src[i], target[i])
                        return False
                    #if DEBUG: print("Ok.")
                return True

            if not check_arr_equal(src_el, target_el):
                print("Verification Failed.")
                sys.exit(1)

    print("Verified.")
    sys.exit(0)

else:
    print("ERROR:  Could not final file at path", FINAL_FILE )
    sys.exit(1)
