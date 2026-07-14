# Transform an EgoRear calibration JSON to match a pad-to-square + resize
# preprocessing of the camera stream (e.g. 1280x720 -> pad to 1280x1280 ->
# resize to 872x872), keeping the full field of view.
#
# Both operations are exact on the Scaramuzza model — no refitting:
#   pad (top/left offset):  image_center += offset
#   uniform resize by s:    image_center *= s
#                           polynomialW2C (theta -> rho px)  *= s
#                           polynomialC2W a_i *= s^(1-i)  (ray (u,v,f(rho));
#                             pixels scale by s, so f must satisfy
#                             f_new(s*rho) = s*f_old(rho))
#
# Usage:
#   python pad_calib_to_square.py my_rig/camera_front_left.json \
#       --out-size 872 -o my_rig_padded/
# Preprocess frames the same way at runtime:
#   pad_lr = (max(W,H)-W)//2, pad_tb = (max(W,H)-H)//2, then resize.

import argparse
import json
import os

import numpy as np


def pad_and_scale_calib(calib, out_size):
    h, w = calib["size"]
    side = max(h, w)
    pad_x = (side - w) // 2
    pad_y = (side - h) // 2
    s = out_size / side

    cx, cy = calib["image_center"]
    c2w = np.asarray(calib["polynomialC2W"], dtype=float)
    w2c = np.asarray(calib["polynomialW2C"], dtype=float)

    out = dict(calib)
    out["size"] = [out_size, out_size]
    out["image_center"] = [(cx + pad_x) * s, (cy + pad_y) * s]
    out["polynomialW2C"] = (w2c * s).tolist()
    out["polynomialC2W"] = [a * s ** (1 - i) for i, a in enumerate(c2w)]
    out["imageCircleRadius"] = int(round(calib.get("imageCircleRadius", side / 2) * s))
    out["name"] = calib.get("name", "") + f"_padded{out_size}"
    return out, (pad_x, pad_y, s)


def _egorear_project(pts, center, w2c):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    norm = np.sqrt(x * x + y * y)
    theta = np.arctan(-z / norm)
    rho = np.polynomial.polynomial.polyval(theta, np.asarray(w2c))
    return np.stack([x / norm * rho + center[0], y / norm * rho + center[1]], axis=1)


def verify(calib, out, pad_x, pad_y, s):
    """Project random 3D points with both calibrations; pixel positions must
    map through the pad+resize exactly."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0.05, 1.2, 2000)
    phi = rng.uniform(-np.pi, np.pi, 2000)
    pts = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                    np.cos(theta)], axis=-1) * 100.0
    uv_orig = _egorear_project(pts, calib["image_center"], calib["polynomialW2C"])
    uv_new = _egorear_project(pts, out["image_center"], out["polynomialW2C"])
    uv_expected = (uv_orig + [pad_x, pad_y]) * s
    return float(np.abs(uv_new - uv_expected).max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("calib_json", nargs="+")
    p.add_argument("--out-size", type=int, default=872)
    p.add_argument("-o", "--out-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.calib_json:
        with open(path) as f:
            calib = json.load(f)
        out, (pad_x, pad_y, s) = pad_and_scale_calib(calib, args.out_size)
        err = verify(calib, out, pad_x, pad_y, s)
        out_path = os.path.join(args.out_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"{os.path.basename(path)}: pad=({pad_x},{pad_y}) scale={s:.5f} "
              f"round-trip error {err:.2e} px -> {out_path}")


if __name__ == "__main__":
    main()
