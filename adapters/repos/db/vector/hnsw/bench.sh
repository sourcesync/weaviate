#!/bin/bash

set -e

export DATASIZE=2000000
export QUERYSIZE=1000
export MULTI=false
export START=2000000
export INCREMENT=990000

go test -run TestBench -v -timeout 0

# BATCH ADD
# // loop for queries, break after 1 iteration if "multi" is false
# 	batch_size := 10000
# 	for size <= data_size {
# 		fmt.Println("loading vectors", curr, ":", size, "to hnsw index...")
# 		t1 := time.Now()
# 		for i := 0; i < len(testVectors); i += batch_size {
# 			fmt.Println("adding vecs:", i, "to", i+batch_size)
# 			err := index.AddBatch(uint64(i), testVectors[i:i+batch_size])
# 			require.Nil(t, err)
# 		}

# NORMAL ADD
# for size <= data_size {
# 		fmt.Println("loading vectors", curr, ":", size, "to hnsw index...")
# 		t1 := time.Now()
# 		for i, vec := range testVectors[curr:size] {
# 			// fmt.Println("adding vecs:", i, "to", i+batch_size)
# 			err := index.Add(uint64(i+curr), vec)
# 			require.Nil(t, err)
# 		}