#!/bin/bash

# This script will help you build the weaivate executable locally (ie, without docker ).
# We don't recommend that you run this unless you know what you are doing with golang.

set -e
set -x

#GOPATH="$HOME/go"
#GOROOT=/usr/local/bin
#export PATH=$GOPATH:$GOROOT/bin:$PATH 

# MAC
export GOPATH=$HOME/go
export GOROOT=/usr/local/opt/go/libexec
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH

#cd .. && go build -o ./weaviate-server -modfile=gsi/go.mod ./cmd/weaviate-server/main.go
cd .. && go build -o ./weaviate-server ./cmd/weaviate-server/main.go
