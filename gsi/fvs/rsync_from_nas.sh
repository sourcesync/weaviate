#!/bin/bash

set -e
set -x

SRC="gwilliams@192.168.99.107:/mnt/nas1/weaviate_benchmark_results/fvs/"

rsync -azvdO $SRC ./results/

