#!/usr/bin/env bash
# Build the benchmark image (cached after first run) and run the speed benchmark.
# Usage: ./benchmark/run.sh [benchmark_speed.py args, e.g. --fp16 --iters 500]
set -e
cd "$(dirname "$0")/.."

docker build -t egorear-bench -f benchmark/Dockerfile benchmark/
docker run --rm --gpus all \
    -v "$(pwd)":/workspace/EgoRear \
    egorear-bench \
    python benchmark/benchmark_speed.py "$@"
