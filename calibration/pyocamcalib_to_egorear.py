# Convert a py-OCamCalib (github.com/jakarto3d/py-OCamCalib) calibration JSON
# into the Scaramuzza calibration format expected by EgoRear
# (pose_estimation/utils/camera_calib_file/*.json).
#
# The two formats differ in more than field names:
#   - py-OCamCalib inverse_poly: HIGHEST-degree-first (np.polyval convention),
#     a function of theta_i = angle from the optical axis (arccos(z/|p|), z forward).
#   - EgoRear polynomialW2C: LOWEST-degree-first, a function of
#     theta_e = atan(-z/sqrt(x^2+y^2)) = theta_i - pi/2  (z forward).
#   So the polynomial must be re-fitted in the shifted variable, not copied.
#   - EgoRear polynomialC2W follows the original MATLAB OCamCalib sign convention
#     (z component of the ray is negative toward the scene) -> negate taylor_coefficient.
#   - py-OCamCalib's stretch_matrix has no counterpart in EgoRear's math (the
#     'affine' field exists in the JSON but is never read by the code), so a
#     strongly non-identity stretch cannot be represented -> warn.
#
# Usage:
#   python pyocamcalib_to_egorear.py <pyocamcalib_calib.json> \
#       --width 1280 --height 960 --name camera_front_left -o <out_dir>
#
# The script self-checks by projecting sample 3D points with both formulas and
# reporting the max pixel disagreement.

import argparse
import json
import os

import numpy as np


def egorear_project(pts3d, w2c_ascending, center, size_hw):
    """Replicates FishEyeCameraCalibratedModel.world2camera_pytorch in numpy."""
    x, y, z = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
    norm = np.sqrt(x * x + y * y)
    theta = np.arctan(-z / norm)
    rho = sum(a * theta**i for i, a in enumerate(w2c_ascending))
    u = x / norm * rho + center[0]
    v = y / norm * rho + center[1]
    return np.stack([u, v], axis=-1)


def pyocamcalib_project(pts3d, inverse_poly_descending, center):
    """Replicates py-OCamCalib Camera.world2cam_fast (without stretch matrix)."""
    x, y, z = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
    r = np.linalg.norm(pts3d, axis=1)
    theta_i = np.arccos(z / r)
    rho = np.polyval(inverse_poly_descending, theta_i)
    norm = np.sqrt(x * x + y * y)
    u = x / norm * rho + center[0]
    v = y / norm * rho + center[1]
    return np.stack([u, v], axis=-1)


def max_incident_angle(taylor_ascending, center, size_hw):
    """Largest angle from the optical axis that still lands on the sensor."""
    h, w = size_hw
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
    rho_max = np.linalg.norm(corners - np.asarray(center), axis=1).max()
    rho = np.linspace(1.0, rho_max, 2000)
    z = np.polyval(np.asarray(taylor_ascending)[::-1], rho)
    theta_i = np.arccos(z / np.sqrt(rho**2 + z**2))
    return float(theta_i.max())


def convert_to_egorear(taylor, inv_poly, center, stretch, width, height, name,
                       degree=8):
    """Convert py-OCamCalib parameters to an EgoRear calibration dict.

    Returns (egorear_dict, report_lines).
    taylor: ascending cam2world coefficients; inv_poly: descending (np.polyval)
    world2cam coefficients in the angle-from-optical-axis variable.
    """
    taylor = np.asarray(taylor, dtype=float)
    inv_poly = np.asarray(inv_poly, dtype=float)
    center = list(map(float, center))
    stretch = np.asarray(stretch if stretch is not None else np.eye(2), dtype=float)
    report = []

    if np.abs(stretch - np.eye(2)).max() > 1e-2:
        report.append(
            f"WARNING: stretch_matrix deviates from identity by "
            f"{np.abs(stretch - np.eye(2)).max():.4f}; EgoRear ignores it, "
            f"expect up to ~that fraction of rho in pixel error.")

    # Refit the world->camera polynomial in EgoRear's shifted angle variable.
    theta_i_max = min(max_incident_angle(taylor, center, (height, width)) + 0.05,
                      np.pi - 1e-3)
    theta_i = np.linspace(1e-3, theta_i_max, 2000)
    rho = np.polyval(inv_poly, theta_i)
    theta_e = theta_i - np.pi / 2.0
    w2c = np.polynomial.polynomial.polyfit(theta_e, rho, deg=degree)  # ascending

    fit_err = np.abs(np.polynomial.polynomial.polyval(theta_e, w2c) - rho)
    report.append(
        f"W2C refit: degree {degree}, max fit error {fit_err.max():.4f} px "
        f"over incident angles 0..{np.degrees(theta_i_max):.1f} deg")

    # Cross-check: project random 3D points with both models.
    rng = np.random.default_rng(0)
    theta_s = rng.uniform(0.05, theta_i_max - 0.05, 5000)
    phi_s = rng.uniform(-np.pi, np.pi, 5000)
    pts3d = np.stack([np.sin(theta_s) * np.cos(phi_s),
                      np.sin(theta_s) * np.sin(phi_s),
                      np.cos(theta_s)], axis=-1) * 100.0
    uv_ego = egorear_project(pts3d, w2c, center, (height, width))
    uv_py = pyocamcalib_project(pts3d, inv_poly, center)
    err = np.linalg.norm(uv_ego - uv_py, axis=1)
    report.append(
        f"cross-check EgoRear-formula vs py-OCamCalib-formula: "
        f"max {err.max():.4f} px, mean {err.mean():.4f} px")

    out = {
        "name": f"{name}_scaramuzza",
        "size": [height, width],
        "imageCircleRadius": int(round(float(np.max(rho)))),
        "image_center": center,
        "polynomialC2W": (-taylor).tolist(),
        "polynomialW2C": w2c.tolist(),
        "affine": [float(stretch[0, 0]), float(stretch[0, 1]), float(stretch[1, 0])],
    }
    return out, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("calib_json", help="py-OCamCalib calibration JSON")
    p.add_argument("--width", type=int, required=True, help="image width in px")
    p.add_argument("--height", type=int, required=True, help="image height in px")
    p.add_argument("--name", required=True,
                   help="EgoRear camera name, e.g. camera_front_left")
    p.add_argument("-o", "--out-dir", default=".")
    p.add_argument("--degree", type=int, default=8,
                   help="degree of the refitted W2C polynomial")
    args = p.parse_args()

    with open(args.calib_json) as f:
        calib = json.load(f)

    out, report = convert_to_egorear(
        calib["taylor_coefficient"], calib["inverse_poly"],
        calib["distortion_center"], calib.get("stretch_matrix"),
        args.width, args.height, args.name, degree=args.degree,
    )
    print("\n".join(report))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
