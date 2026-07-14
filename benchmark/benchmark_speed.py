# Speed benchmark for the EgoRear inference pipeline (no trained weights needed —
# latency does not depend on weight values, so the model is randomly initialized).
#
# Times three nested stages at batch size 1 (the real-time dual-camera case):
#   1. stage1  : stereo 2D heatmap estimator (ResNet18 + FPN + conv head)
#   2. stage12 : stage 1 + multi-view heatmap refinement (MVFEX + JQA)
#   3. full    : stages 1+2 + 3D joint estimator (transformer lifting)

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import yaml


def build_model(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]["init_args"]["model_cfg"]

    from pose_estimation.models.estimator import EgoPoseFormerMVFEX

    model = EgoPoseFormerMVFEX(**model_cfg)
    return model, model_cfg


def make_inputs(batch, num_views, image_size, device):
    img = torch.randn(batch, num_views, 3, image_size[0], image_size[1], device=device)

    # Plausible device->camera extrinsics (identity rotation, ~6 cm baseline, meters).
    coord_trans_mat = torch.eye(4, device=device).repeat(batch, num_views, 1, 1)
    offsets = [0.06, -0.06, 0.06, -0.06]
    for v in range(num_views):
        coord_trans_mat[:, v, 0, 3] = offsets[v]

    return img, coord_trans_mat


@torch.no_grad()
def time_fn(fn, iters, warmup, autocast_dtype=None):
    def run():
        if autocast_dtype is not None:
            with torch.autocast("cuda", dtype=autocast_dtype):
                fn()
        else:
            fn()

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    times_ms = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def report(name, times_ms):
    mean = statistics.mean(times_ms)
    p50 = statistics.median(times_ms)
    p95 = sorted(times_ms)[int(len(times_ms) * 0.95) - 1]
    print(
        f"{name:<38s} mean {mean:7.2f} ms | p50 {p50:7.2f} ms | "
        f"p95 {p95:7.2f} ms | {1000.0 / mean:6.1f} FPS"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="./configs/ego4view_rw_pose3d_stereo_front.yaml"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--fp16", action="store_true", help="run under autocast fp16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required (the camera model allocates on cuda).")

    torch.backends.cudnn.benchmark = True

    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"torch   : {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"config  : {args.config}")
    print(f"batch   : {args.batch} | iters {args.iters} | warmup {args.warmup} "
          f"| precision {'fp16-autocast' if args.fp16 else 'fp32'}")

    model, model_cfg = build_model(args.config)
    model = model.cuda().eval()

    num_views = model_cfg["num_views"]
    image_size = model_cfg["image_size"]
    img, coord_trans_mat = make_inputs(args.batch, num_views, image_size, "cuda")
    print(f"input   : img {list(img.shape)}, coord_trans_mat {list(coord_trans_mat.shape)}")
    print()

    dtype = torch.float16 if args.fp16 else None

    stage1 = model.heatmap_estimator.heatmap_estimator_stereo_front
    report(
        "stage 1 (2D heatmap estimator)",
        time_fn(lambda: stage1(img), args.iters, args.warmup, dtype),
    )
    report(
        "stages 1+2 (+ MV heatmap refinement)",
        time_fn(lambda: model.heatmap_estimator(img), args.iters, args.warmup, dtype),
    )
    report(
        "full pipeline (+ 3D estimator)",
        time_fn(
            lambda: model(img, coord_trans_mat, None), args.iters, args.warmup, dtype
        ),
    )

    print()
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")


if __name__ == "__main__":
    main()
