#!/bin/bash

set -e

export DATASIZE=500000000
export QUERYSIZE=1000
export INCREMENT=50000000
export START=100000000

go test -run TestBench -v -timeout 0