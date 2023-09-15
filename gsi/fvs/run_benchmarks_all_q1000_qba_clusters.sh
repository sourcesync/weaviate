#!/bin/bash

set -x
set -e

./run_benchmarks_deep1M_q1000_qbq_clusters.sh
./run_benchmarks_deep2M_q1000_qbq_clusters.sh
./run_benchmarks_deep5M_q1000_qbq_clusters.sh
./run_benchmarks_deep10M_q1000_qbq_clusters.sh
./run_benchmarks_deep20M_q1000_qbq_clusters.sh
./run_benchmarks_deep50M_q1000_qbq_clusters.sh
#./run_benchmarks_deep100M_q1000_qbq_clusters.sh
#./run_benchmarks_deep250M_q1000_qbq_clusters.sh
#./run_benchmarks_deep500M_q1000_qbq_clusters.sh
