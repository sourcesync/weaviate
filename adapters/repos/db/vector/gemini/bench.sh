#!/bin/bash

set -e
set -x

#
# Golang env vars
#

# LINUX
export GOPATH="$HOME/go"
export GOROOT=/usr/local/go
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH
# MAC
#export GOPATH=$HOME/go
#export GOROOT=/usr/local/opt/go/libexec
#export PATH=$GOPATH/bin:$GOROOT/bin:$PATH

#
# Gemini Plugin env vars
#
export GEMINI_ALLOCATION_ID=fd283b38-3e4a-11eb-a205-7085c2c5e516  # apu11=fd283b38-3e4a-11eb-a205-7085c2c5e516 # apu12=0b391a1a-b916-11ed-afcb-0242ac1c0002
export GEMINI_DATA_DIRECTORY=/home/public #/var/lib/weaviate
export GEMINI_FVS_SERVER=localhost
export GEMINI_DEBUG=false
export GEMINI_MIN_RECORDS_CHECK=true

#
# Env vars for the benchmark/test program
#
export DSET="deep1b"
export DATAPATH="/home/public/deep-500M.npy"
export NUMRECS=500000000
export DIM=96
export SEARCH="clusters"
export QUERYPATH="/home/public/deep-queries.npy"
export CSVPATH="/home/gwilliams/Projects/Weaviate/results/algo_direct/"
export BITS=512

# Run with Golang debugger DLV
# DLV is a Golang debugger that will need to have installed locally
#which dlv
#cd .. && dlv debug cmd/weaviate-server/main.go --build-flags -modfile=gsi/go.mod -- --host=0.0.0.0 --port=8081 --scheme=http
#dlv debug cmd/weaviate-server/main.go  -- --host=0.0.0.0 --port=8091 --scheme=http

# Run with Golang
go test -run TestBench -v -timeout 0
