import os
import h5py
import json
import time
import subprocess

def write_queries_to_json(fp):
    with h5py.File(fp, 'r') as f:
        test_vectors = f["test"]
        vector_write_array = []
        for vector in test_vectors:
            vector_write_array.append(vector.tolist())
        with open("queries.json", "w", encoding="utf-8") as jf:
            json.dump(vector_write_array, jf, indent=2)

def run_benchmarks():
    