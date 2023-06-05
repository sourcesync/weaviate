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
BASE_ATLAS = "/mnt/nas1/atlas_data/benchmarking/base_atlas.npy"

BASE_ATLAS_1M = "/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_1M.npy"
BASE_ATLAS_2M = "/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_2M.npy"
BASE_ATLAS_5M = "/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_5M.npy"
BASE_ATLAS_10M = "/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_10M.npy"
BASE_ATLAS_20M = "/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_20M.npy"
BASE_ATLAS_30M = "/mnt/nas1/atlas_data/benchmarking/sets/base_atlas_30M.npy"

# load base atlas embeddings
base = np.load(BASE_ATLAS, allow_pickle=True)

# CONSTANTS
oneM = 1000000
twoM = 2000000
fiveM = 5000000
tenM = 10000000
twentyM = 20000000
thirtyM = 30000000

# 1M
print("Slicing 1M ...")
atlas_1M = base[:oneM]
print("Saving 1M ...")
np.save(BASE_ATLAS_1M, atlas_1M)
print("Done with 1M!")

# 2M
print("Slicing 2M ...")
atlas_2M = base[:twoM]
print("Saving 2M ...")
np.save(BASE_ATLAS_2M, atlas_2M)
print("Done with 2M!")

# 5M
print("Slicing 5M ...")
atlas_5M = base[:fiveM]
print("Saving 5M ...")
np.save(BASE_ATLAS_5M, atlas_5M)
print("Done with 5M!")

# 10M
print("Slicing 10M ...")
atlas_10M = base[:tenM]
print("Saving 10M ...")
np.save(BASE_ATLAS_10M, atlas_10M)
print("Done with 10M!")

# 20M
print("Slicing 20M ...")
atlas_20M = base[:twentyM]
print("Saving 20M ...")
np.save(BASE_ATLAS_20M, atlas_20M)
print("Done with 20M!")

# 30M
print("Slicing 30M ...")
atlas_30M = base[:thirtyM]
print("Saving 30M ...")
np.save(BASE_ATLAS_30M, atlas_30M)
print("Done with 30M!")

