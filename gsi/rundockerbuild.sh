#!/bin/bash

# This script is for running the docker build for GSI's weaivate container

set -e
set -x

cd .. && DOCKER_BUILDKIT=0 docker build -f gsi/docker/Dockerfile.gemini -t gsi/weaviate .
