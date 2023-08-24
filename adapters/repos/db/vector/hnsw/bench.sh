#!/bin/bash

set -e

export DATASIZE="250000000 500000000 1000000000" #MUST BE STRING
export QUERYSIZE=1000
export CSVPATH="/home/jbenson/bench/redo/"

go test -run TestBench -v -timeout 0 | tee -a /home/jbenson/bench_logs_last.txt
