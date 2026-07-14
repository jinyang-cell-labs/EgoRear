# Live EgoRear skeleton demo for a custom front-stereo fisheye rig.
#
# Pipeline: two V4L2 cameras -> pad frames to square (keeps full FOV, matches
# the padded calibration JSONs from calibration/pad_calib_to_square.py)
# -> EgoPoseFormerMVFEX (pretrained ego4view_rw_pose3d_stereo_front)
# -> streams to a rerun viewer: per-view images with 2D heatmap peaks and
# reprojected 3D joints, plus the animated 3D skeleton.
#
# Run inside the egorear-bench container (see demo/run_live_demo.sh); the
# rerun viewer runs on the host.

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import rerun as rr

from pose_estimation.models.estimator import EgoPoseFormerMVFEX
from pose_estimation.utils.loss import get_max_preds
from pose_estimation.utils.state_dict import fix_model_state_dict

# Skeleton topology from pose_estimation/utils/skeleton.py (not imported —
# that module pulls in open3d, which the container doesn't need).
JOINT_NAMES = ["head", "neck", "upperarm_l", "upperarm_r", "lowerarm_l",
               "lowerarm_r", "hand_l", "hand_r", "thigh_l", "thigh_r",
               "calf_l", "calf_r", "foot_l", "foot_r", "ball_l", "ball_r"]
BONES = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
         (2, 8), (3, 9), (8, 10), (9, 11), (10, 12), (11, 13),
         (12, 14), (13, 15), (8, 9)]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(config_path, ckpt_path, calib_dir):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]["init_args"]["model_cfg"]
    model_cfg["pose3d_cfg"]["camera_calib_file_dir_path"] = calib_dir

    model = EgoPoseFormerMVFEX(**model_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    state = fix_model_state_dict(state, rm_name="network._orig_mod.")
    state = fix_model_state_dict(state, rm_name="network.")
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def load_extrinsics(path):
    with open(path) as f:
        ext = json.load(f)
    m = ext["coord_transformation_matrix"]
    ctm = np.stack([np.asarray(m["device_to_camera_front_left"]),
                    np.asarray(m["device_to_camera_front_right"])])
    return torch.tensor(ctm, dtype=torch.float32, device="cuda").unsqueeze(0)


def load_calibs(calib_dir):
    calibs = []
    for name in ("camera_front_left", "camera_front_right"):
        with open(os.path.join(calib_dir, f"{name}.json")) as f:
            calibs.append(json.load(f))
    return calibs


def pad_to_square(frame):
    h, w = frame.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    return cv2.copyMakeBorder(frame, top, side - h - top, left, side - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def preprocess(frame_sq):
    img = cv2.resize(frame_sq, (256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1))


def reproject(joints_cm, ctm_4x4, calib, display_size):
    """Model output joints (16,3 cm, device frame) -> pixels in the padded
    square display image."""
    pts_m = np.hstack([joints_cm * 0.01, np.ones((len(joints_cm), 1))])
    pts_cam = (ctm_4x4 @ pts_m.T).T[:, :3]  # meters, camera frame
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    norm = np.maximum(np.sqrt(x * x + y * y), 1e-9)
    theta = np.arctan(-z / norm)
    rho = np.polynomial.polynomial.polyval(theta, np.asarray(calib["polynomialW2C"]))
    u = x / norm * rho + calib["image_center"][0]
    v = y / norm * rho + calib["image_center"][1]
    s = display_size / calib["size"][0]
    return np.stack([u, v], axis=1) * s


def open_camera(dev, width, height):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"cannot open {dev}")
    return cap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", default="./configs/ego4view_rw_pose3d_stereo_front.yaml")
    p.add_argument("--calib-dir", default="./calibration/my_rig_872")
    p.add_argument("--extrinsics", default="./calibration/my_rig_872/extrinsics.json")
    p.add_argument("--left", default="/dev/video2")
    p.add_argument("--right", default="/dev/video4")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--display-size", type=int, default=512)
    p.add_argument("--connect", default="rerun+http://127.0.0.1:9876/proxy",
                   help="rerun viewer gRPC url, or 'save:<file.rrd>'")
    p.add_argument("--max-frames", type=int, default=0, help="0 = run forever")
    args = p.parse_args()

    rr.init("egorear-live", spawn=False)
    if args.connect.startswith("save:"):
        rr.save(args.connect[5:])
    else:
        rr.connect_grpc(args.connect)

    print("loading model ...")
    model = load_model(args.config, args.ckpt, args.calib_dir)
    ctm = load_extrinsics(args.extrinsics)
    calibs = load_calibs(args.calib_dir)
    print("model ready")

    caps = [open_camera(args.left, args.width, args.height),
            open_camera(args.right, args.width, args.height)]

    ds = args.display_size
    n = 0
    t_last = time.time()
    while True:
        frames = []
        for cap in caps:
            ok, f = cap.read()
            if not ok:
                sys.exit("frame grab failed")
            frames.append(f)

        squares = [pad_to_square(f) for f in frames]
        img = torch.stack([preprocess(s) for s in squares]).unsqueeze(0).cuda()

        with torch.no_grad():
            t0 = time.time()
            list_pose, list_heatmap = model(img, ctm, None)
            torch.cuda.synchronize()
            dt_ms = (time.time() - t0) * 1000

        joints_cm = list_pose[-1][0].cpu().numpy()          # (16,3) device frame
        heatmap = list_heatmap[-1][0]                       # (2,15,H,W)
        pts2d, maxvals, _ = get_max_preds(heatmap, threshold=0.3, normalize=True)

        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("frame", n)
        else:
            rr.set_time("frame", sequence=n)
        joints_m = joints_cm * 0.01
        rr.log("skeleton/joints", rr.Points3D(joints_m, radii=0.015))
        rr.log("skeleton/bones", rr.LineStrips3D(
            [[joints_m[a], joints_m[b]] for a, b in BONES]))

        for v, side in enumerate(("left", "right")):
            disp = cv2.resize(squares[v], (ds, ds))
            rr.log(f"cam/{side}/image",
                   rr.Image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=70))
            # heatmap peaks: normalized [0,1] -> display px (skip low-confidence)
            conf = maxvals[v].cpu().numpy().ravel()
            peaks = pts2d[v].cpu().numpy() * ds
            rr.log(f"cam/{side}/heatmap_peaks",
                   rr.Points2D(peaks[conf > 0.3], colors=(255, 220, 0), radii=4))
            # reprojected 3D joints
            uv = reproject(joints_cm, ctm[0, v].cpu().numpy(), calibs[v], ds)
            rr.log(f"cam/{side}/reprojected_3d",
                   rr.Points2D(uv, colors=(0, 200, 255), radii=3))

        n += 1
        if n % 30 == 0:
            fps = 30.0 / (time.time() - t_last)
            t_last = time.time()
            print(f"frame {n}: {fps:.1f} fps end-to-end, model {dt_ms:.0f} ms")
        if args.max_frames and n >= args.max_frames:
            break


if __name__ == "__main__":
    main()
