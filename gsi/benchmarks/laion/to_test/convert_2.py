import torch
import os
import numpy as np
import pickle

# load tensor
#dir = "/mnt/nas1/atlas_data/indices/atlas/wiki/xl"
dir = "/mnt/nas1/atlas_data/benchmarking/npy_test"

for filename in os.listdir(dir):
    # only test on embeddings0.npy
    if "embeddings0" in filename:
        fp = os.path.join(dir, filename)
        print("filename: ", filename)

        try:
            """# load tensor
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
            print("\n done!")"""

            # memmap
            # load file

            # load from embeddings.npy
            npy = np.load(fp)

            print("Transposing...")
            arr = np.transpose(npy)
            shape = arr.shape
            print("Shape= ",shape,"dtype= ", arr.dtype)

            # astype
            arr1 = arr.astype(np.float32)
            print("astype: ", arr1.dtype)

            # a new file to store all
            fp_map = "/mnt/nas1/atlas_data/benchmarking/atlas_test_0531.npy"

            # create memmap
            fp_mmap = np.memmap(fp_map, dtype='float32', mode='w+', shape=shape)
            print("created a new mmap")
            # store arr to mmap
            fp_mmap[:] = arr1[:]
            print("t/f? ", fp_mmap.filename == os.path.abspath(fp_map))

            fp_mmap.flush()
            newfp = np.memmap(fp_map, dtype='float32', mode='r', shape=shape)


        except:
            # now try as pickled object
            print("Torch load failed.  Trying to load as a pickled object...")
            with open(fp, 'rb') as f:
                obj = f.read()
                data = pickle.loads(obj, encoding='latin1')
                print( "Got data->", type(data) )

