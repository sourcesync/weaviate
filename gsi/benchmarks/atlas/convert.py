import torch
import os
import numpy as np

# load tensor
dir = "/mnt/nas1/atlas_data/indices/atlas/wiki/xl"

for filename in os.listdir(dir):
    fp = os.path.join(dir, filename)
    print("filename: ", filename)

    # load tensor
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
