#
# Standard imports
#
import os
import sys
import struct

#
# Installed/external packages
#
import numpy as np


# File Path
BASE_LAION = "/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M.npy"

BASE_LAION_10K = "/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_10K.npy"
BASE_LAION_1M = "/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_1M.npy"
BASE_LAION_2M = "/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_2M.npy"
BASE_LAION_5M = "/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_5M.npy"
BASE_LAION_10M = "/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_10M.npy"
BASE_LAION_20M = "/mnt/nas1/laion400m/benchmarking/sets_nor/base_laion24M_20M.npy"


# load base atlas embeddings
print("loading base LAION ... ")
base = np.load(BASE_LAION, allow_pickle=True)

# CONSTANTS
tenK = 10000
oneM = 1000000
twoM = 2000000
fiveM = 5000000
tenM = 10000000
twentyM = 20000000
#thirtyM = 30000000

# 10K
print("Slicing 10K ...")
slice_10K = base[:tenK]
print("Saving 10K ...")
np.save(BASE_LAION_10K, slice_10K)
print("Done with 10K!")

# 1M
print("Slicing 1M ...")
slice_1M = base[:oneM]
print("Saving 1M ...")
np.save(BASE_LAION_1M, slice_1M)
print("Done with 1M!")

# 2M
print("Slicing 2M ...")
slice_2M = base[:twoM]
print("Saving 2M ...")
np.save(BASE_LAION_2M, slice_2M)
print("Done with 2M!")

# 5M
print("Slicing 5M ...")
slice_5M = base[:fiveM]
print("Saving 5M ...")
np.save(BASE_LAION_5M, slice_5M)
print("Done with 5M!")

# 10M
print("Slicing 10M ...")
slice_10M = base[:tenM]
print("Saving 10M ...")
np.save(BASE_LAION_10M, slice_10M)
print("Done with 10M!")

# 20M
print("Slicing 20M ...")
slice_20M = base[:twentyM]
print("Saving 20M ...")
np.save(BASE_LAION_20M, slice_20M)
print("Done with 20M!")

