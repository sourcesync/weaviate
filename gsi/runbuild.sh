#!/bin/bash

# This script will help you build the weaivate executable locally (ie, without docker ).
# We don't recommend that you run this unless you know what you are doing with golang.

set -e
set -x

# Standard Weaviate env vars
export OPENAI_APIKEY
export QUERY_DEFAULTS_LIMIT
export AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED
export PERSISTENCE_DATA_PATH
export DEFAULT_VECTORIZER_MODULE
export CLUSTER_HOSTNAME
export ENABLE_MODULES
export TRANSFORMERS_INFERENCE_API

# New Weaviate env vars
export MODULES_PATH
export DEFAULT_VECTOR_INDEX_TYPE

# Gemini Plugin env vars
export GEMINI_ALLOCATION_ID
export GEMINI_DATA_DIRECTORY
export GEMINI_FVS_SERVER
export GEMINI_DEBUG
export GEMINI_MIN_RECORDS_CHECK
export GOPATH
export PATH

# Golang env vars

# This will remove your weaviate data!!
sudo mkdir -p /var/lib/weaviate2
sudo chmod ugo+rw /var/lib/weaviate2
rm -fr /var/lib/weaviate2/*

# LINUX
export GOPATH="$HOME/go"
export GOROOT=/usr/local/go
export PATH=$GOPATH/bin:$GOROOT/bin:$PATH 

# MAC
#export GOPATH=$HOME/go
#export GOROOT=/usr/local/opt/go/libexec
#export PATH=$GOPATH/bin:$GOROOT/bin:$PATH

source .env

#cd .. && go build -o ./weaviate-server -modfile=gsi/go.mod ./cmd/weaviate-server/main.go
cd .. && go build -o ./weaviate-server ./cmd/weaviate-server/main.go  
./weaviate-server  --host=0.0.0.0 --port=8091 --scheme=http
