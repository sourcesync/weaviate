#!/bin/bash

set -e
set -x

TARGET="gwilliams@192.168.99.107:/mnt/nas1/weaviate_benchmark_results/fvs/$(hostname)/all_09092023/"

rsync -azvdO results/qbq "$TARGET"
