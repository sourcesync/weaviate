import torch
import os
import numpy as np
import pickle

# load tensor
dir = "/mnt/nas1/atlas_data/indices/atlas/wiki/xl"

for filename in os.listdir(dir):

    # check if we already created the export
    npy_name = filename.replace('.', '')[:-2] + ".npy"
    npy_dir = "/mnt/nas1/atlas_data/benchmarking/npy"
    npy_fp = os.path.join(npy_dir, npy_name)
    if os.path.exists(npy_fp):
        print("Already processed", filename)
        continue
        
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
        npy_dir = "/mnt/nas1/atlas_data/benchmarking/npy"
        npy_fp = os.path.join(npy_dir, npy_name)
        np.save(npy_fp, npy)
        print(npy_fp)
        print("\n done!")
    except:
        # now try as pickled object
        print("Torch load failed.  Trying to load as a pickled object...")
        with open(fp, 'rb') as f:
            obj = f.read()
            data = pickle.loads(obj, encoding='latin1')
            print( "Got data->", type(data) )

