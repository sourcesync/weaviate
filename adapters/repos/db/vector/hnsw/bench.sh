#!/bin/bash

set -e

export DATANAME=10K
export DATASIZE=10000
export QUERYSIZE=1000

go test -run TestBench -v -timeout 0