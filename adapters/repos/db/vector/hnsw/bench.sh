#!/bin/bash

set -e

export DATASIZE=1000000
export QUERYSIZE=1000
export INCREMENT=990000
export START=10000

go test -run TestBench -v -timeout 0