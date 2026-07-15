# Run the pretrained checkpoint on an Ego4View-RW dataset sequence (the
# paper's own test data) and stream prediction vs ground truth to rerun.
# This shows the algorithm's best-case behavior: correct cameras, correct
# mounting, in-distribution appearance. Compare against the live demo on
# your own rig to separate "domain gap" from "pipeline bugs".
#
# Usage (inside the egorear-bench container):
#   python demo/dataset_demo.py --seq data_subset/2024_09_17/S13/seq_2-6 \
#       --ckpt pretrained/ego4view_rw_pose3d_stereo_front/.../epoch=11.ckpt \
#       --connect save:demo/dataset_eval.rrd

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rerun as rr

from live_demo import (BONES, load_model, preprocess, reproject)

LIST_JOINTS = ["Head", "Neck", "LeftArm", "RightArm", "LeftForeArm",
               "RightForeArm", "LeftHand", "RightHand", "LeftUpLeg",
               "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot",
               "LeftToeBase", "RightToeBase"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", required=True, help="path to a sequence dir")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="./configs/ego4view_rw_pose3d_stereo_front.yaml")
    p.add_argument("--calib-dir", default="./pose_estimation/utils/camera_calib_file/ego4view")
    p.add_argument("--connect", default="rerun+http://127.0.0.1:9876/proxy")
    p.add_argument("--display-size", type=int, default=512)
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    rr.init("egorear-dataset-eval", spawn=False)
    if args.connect.startswith("save:"):
        rr.save(args.connect[5:])
    else:
        rr.connect_grpc(args.connect)

    model = load_model(args.config, args.ckpt, args.calib_dir)

    seq = args.seq.rstrip("/")
    meta_path = seq.split("-")[0] + "_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    m = meta["coord_transformation_matrix"]
    ctm_np = np.stack([np.asarray(m["device_to_camera_front_left"]),
                       np.asarray(m["device_to_camera_front_right"])])
    ctm = torch.tensor(ctm_np, dtype=torch.float32, device="cuda").unsqueeze(0)

    calibs = []
    for name in ("camera_front_left", "camera_front_right"):
        with open(os.path.join(args.calib_dir, f"{name}.json")) as f:
            calibs.append(json.load(f))

    frames = sorted(glob.glob(os.path.join(seq, "json_smplx", "frame_*.json")),
                    key=lambda p: int(re.search(r"frame_(\d+)", p).group(1)))
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"{len(frames)} frames in {seq}")

    ds = args.display_size
    all_err = []
    for i, jf in enumerate(frames):
        with open(jf) as f:
            data = json.load(f)
        gt = np.array([data["joints"][j]["device_pts3d"] for j in LIST_JOINTS])

        imgs = []
        views = []
        skip = False
        for cam in ("camera_front_left", "camera_front_right"):
            img_path = jf.replace("json_smplx", f"fisheye_rgb/{cam}").replace(".json", ".png")
            if not os.path.exists(img_path):
                img_path = img_path.replace(".png", ".jpg")
            frame = cv2.imread(img_path)
            if frame is None:
                skip = True
                break
            views.append(frame)
            imgs.append(preprocess(frame))  # already square 872x872
        if skip:
            continue
        img = torch.stack(imgs).unsqueeze(0).cuda()

        with torch.no_grad():
            poses, heatmaps = model(img, ctm, None)
        pred = poses[-1][0].cpu().numpy()  # (16,3) cm

        err_mm = np.linalg.norm(pred - gt, axis=1).mean() * 10.0
        all_err.append(err_mm)

        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("frame", i)
        else:
            rr.set_time("frame", sequence=i)
        rr.log("mpjpe_mm", rr.Scalars(err_mm) if hasattr(rr, "Scalars") else rr.Scalar(err_mm))
        for label, joints, color in (("pred", pred, (0, 200, 255)),
                                     ("gt", gt, (0, 255, 0))):
            jm = joints * 0.01
            rr.log(f"skeleton/{label}/joints", rr.Points3D(jm, radii=0.015, colors=color))
            rr.log(f"skeleton/{label}/bones", rr.LineStrips3D(
                [[jm[a], jm[b]] for a, b in BONES], colors=color))
        for v, side in enumerate(("left", "right")):
            disp = cv2.resize(views[v], (ds, ds))
            rr.log(f"cam/{side}/image",
                   rr.Image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=70))
            uv_pred = reproject(pred, ctm_np[v], calibs[v], ds)
            uv_gt = reproject(gt, ctm_np[v], calibs[v], ds)
            rr.log(f"cam/{side}/pred_2d", rr.Points2D(uv_pred, colors=(0, 200, 255), radii=3))
            rr.log(f"cam/{side}/gt_2d", rr.Points2D(uv_gt, colors=(0, 255, 0), radii=3))

        if (i + 1) % 50 == 0:
            print(f"frame {i+1}/{len(frames)}: running MPJPE {np.mean(all_err):.1f} mm")

    print(f"\nsequence MPJPE: {np.mean(all_err):.1f} mm over {len(all_err)} frames "
          f"(paper reports ~50-70 mm for rw stereo_front)")


if __name__ == "__main__":
    main()
