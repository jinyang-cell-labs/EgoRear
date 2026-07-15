#!/usr/bin/env bash
# Open the 17s Ego4View-RW test-sequence evaluation (prediction vs ground
# truth) in the rerun viewer. Regenerates the recording first if missing.
set -euo pipefail
cd "$(dirname "$0")/.."

RRD="demo/dataset_eval.rrd"
VIEWER="$HOME/repo/py-OCamCalib/.venv/bin/rerun"

if [ ! -f "$RRD" ]; then
    echo "[prep] recording missing — running the evaluation (~2 min) ..."
    docker run --rm --gpus all -v "$(pwd)":/workspace/EgoRear egorear-bench \
        python demo/dataset_demo.py \
        --seq data_subset/2024_09_17/S13/seq_2-6 \
        --ckpt "pretrained/ego4view_rw_pose3d_stereo_front/lightning_logs/version_0/checkpoints/epoch=11.ckpt" \
        --connect "save:/workspace/EgoRear/$RRD"
fi

echo "[viewer] green = ground truth, cyan = prediction."
echo "[viewer] press play (or drag the 'frame' timeline) to animate."
exec "$VIEWER" "$RRD"
