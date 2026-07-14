# Benchmark environment for EgoRear.
# torch 2.4.0 + cu121 is pinned because prebuilt mmcv 2.2.0 wheels
# (MultiScaleDeformableAttn CUDA op) only exist up to this combination.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# system libs required by opencv-python (pulled in by mmcv)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        lightning pytorch-lightning \
        natsort loguru einops timm pyyaml \
        opencv-python-headless pillow scipy \
        'jsonargparse[signatures]>=4.27.7' && \
    pip install --no-cache-dir mmcv==2.2.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# live-demo visualization (streams to a rerun viewer on the host)
RUN pip install --no-cache-dir rerun-sdk

WORKDIR /workspace/EgoRear
