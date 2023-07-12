#!/bin/bash

set -e

export DATASIZE=1000000
export QUERYSIZE=1000
export MULTI=false
export START=1000000
export INCREMENT=990000
export CSVPATH="/home/jacob/bench/batch6/"

go test -run TestBench -v -timeout 0
