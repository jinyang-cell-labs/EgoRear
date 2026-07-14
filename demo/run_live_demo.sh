#!/usr/bin/env bash
# Launch the live skeleton demo in the egorear-bench container.
# Start the viewer on the host first:  ~/repo/py-OCamCalib/.venv/bin/rerun
# Usage: ./demo/run_live_demo.sh --ckpt <path-inside-repo> [live_demo.py args]
set -euo pipefail
cd "$(dirname "$0")/.."

docker build -q -t egorear-bench -f benchmark/Dockerfile benchmark/ >/dev/null

exec docker run --rm --gpus all --net=host \
    --device /dev/video0 --device /dev/video2 --device /dev/video4 \
    -v "$(pwd)":/workspace/EgoRear \
    egorear-bench \
    python demo/live_demo.py "$@"
