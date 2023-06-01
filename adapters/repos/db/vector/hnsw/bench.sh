#!/bin/bash

set -e

export DATASIZE=10000
export QUERYSIZE=1000
export MULTI=false
export INCREMENT=990000
export START=10000

go test -run TestBench -v -timeout 0