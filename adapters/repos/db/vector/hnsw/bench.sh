#!/bin/bash

set -e

export DATASIZE="1000000 2000000" #MUST BE STRING
export QUERYSIZE=1000
export CSVPATH="/home/jacob/bench/"

go test -run TestBench -v -timeout 0 | tee -a /home/jacob/logs.txt
