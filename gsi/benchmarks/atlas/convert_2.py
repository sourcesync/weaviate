import torch
import os
import numpy as np
import pickle

# load tensor
dir = "/mnt/nas1/atlas_data/indices/atlas/wiki/xl"
i = 0

for filename in os.listdir(dir):
    while i < 2:
        fp = os.path.join(dir, filename)
        print("filename: ", filename)

        try:
            # load tensor
            print("Trying to load as torch tensor...")
            tensor = torch.load(fp, map_location=torch.device('cpu'))
            print("done loading tensor")

            # convert to np array
            npy = tensor.cpu().numpy()
            print("done converting to npy")

            # save to .npy
            npy_name = filename.replace('.', '')[:-2] + ".npy"
            npy_dir = "/mnt/nas1/atlas_data/benchmarking/npy_test"
            npy_fp = os.path.join(npy_dir, npy_name)
            np.save(npy_fp, npy)
            print(npy_fp)
            print("\n done!")

            # memmap
            # load file

            print("Transposing...")
            arr = np.transpose(npy)
            shape = arr.shape
            print("Shape= ",shape,"dtype= ",arr.dtype)

            #arr = npy
            #shape = arr.shape

            # a new file to store all
            fp = "/mnt/nas1/atlas_data/benchmarking/atlas_test_2.npy"

            # create memmap
            fp_mmap = np.memmap(fp, dtype='float16', mode='w+', shape=shape)
            print("created a new mmap")
            # store arr to mmap
            fp_mmap[:] = arr[:]
            print("t/f? ", fp_mmap.filename == os.path.abspath(fp))

            fp_mmap.flush()
            newfp = np.memmap(fp, dtype='float16', mode='r', shape=shape)
 

        except:
            # now try as pickled object
            print("Torch load failed.  Trying to load as a pickled object...")
            with open(fp, 'rb') as f:
                obj = f.read()
                data = pickle.loads(obj, encoding='latin1')
                print( "Got data->", type(data) )

        print("i: ", i)
        i += 1