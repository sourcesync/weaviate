#!/bin/bash

set -e
set -x

#go env
#echo ""
#export GOPATH="`pwd`/..:/Users/gwilliams/go"
#export GOROOT="`pwd`/..:/usr/local/Cellar/go/1.20.1/libexec"
#go env

go build -cover -o /tmp/test test.go

GOCOVERDIR=coverdata /tmp/test

go tool covdata percent -i=coverdata
