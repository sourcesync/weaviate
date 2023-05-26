import os
import numpy as np

# Directory of converted files
NPY_DIR = "/mnt/nas1/atlas_data/benchmarking/npy"

# Atlas directory of original files
ATLAS_DIR = "/mnt/nas1/atlas_data/indices/atlas/wiki/xl"

i = 0
total_rec = 0

for filename in os.listdir(NPY_DIR):
    print(filename)

    fp = os.path.join(NPY_DIR, filename)
    arr = np.load(fp)
    n, d = arr.shape
    print("n: {}. d: {}".format(n, d) )
    i += 1
    total_rec += d

print(i)
print('total records: ', total_rec)