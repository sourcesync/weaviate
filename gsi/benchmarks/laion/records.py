import os
import numpy as np

# Directory of converted files
NPY_DIR = "/mnt/nas1/laion400m/benchmarking/npy"

# Atlas directory of original files
LAION_DIR = "/mnt/nas1/laion400m/images/img_emb"

i = 0
total_rec = 0

for filename in os.listdir(LAION_DIR):
    print(filename)

    fp = os.path.join(LAION_DIR, filename)
    arr = np.load(fp)
    n, d = arr.shape
    print("n: {}. d: {}".format(n, d) )
    i += 1
    total_rec += n

print(i)
print('total records: ', total_rec)