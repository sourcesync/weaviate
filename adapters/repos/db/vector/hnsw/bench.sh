#!/bin/bash

set -e

export DATANAME=50M
export DATASIZE=50000000
export QUERYSIZE=1000

go test -run TestBench -v -timeout 0

sleep 30

export DATANAME=50M
export DATASIZE=50000000
export QUERYSIZE=1000

go test -run TestBench -v -timeout 0
