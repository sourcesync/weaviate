#!/bin/bash

set -e

export DATASIZE="50000000" #MUST BE STRING
export QUERYSIZE=1000
export CSVPATH="/home/jbenson/bench/"

go test -run TestBench -v -timeout 0
