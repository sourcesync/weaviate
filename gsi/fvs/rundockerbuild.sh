#!/bin/bash

# This script is for running the docker build for GSI's fake vs container.
# Note:  We may end up removing this and re-factoring the docker build if/when we merge with main weaviate

DOCKER_BUILDKIT=0 docker build -f Dockerfile.fake_fvs -t gsi/fake_fvs .
