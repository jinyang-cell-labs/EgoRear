# Stereo extrinsic calibration for EgoRear from paired chessboard views.
#
# No pinhole model is involved anywhere: the Scaramuzza intrinsics (EgoRear
# calibration JSON) map detected corner pixels to 3D rays exactly; PnP runs
# on ray-normalized coordinates with an identity camera matrix.
#
# Pipeline per synchronized image pair (both cameras see the board):
#   corners -> rays (cam2world polynomial) -> solvePnP (planar, IPPE)
#   -> per-pair relative transform T_left<-right
# then median-based outlier rejection (kills chessboard 180-degree-flip
# pairs and motion-desync pairs), averaging, and a symmetric "device" frame
# (midpoint position, slerp-half orientation) for EgoRear's
# device->camera matrices (saved in METERS, EgoRear convention).

import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def load_intrinsics(path):
    with open(path) as f:
        calib = json.load(f)
    return {
        "center": np.asarray(calib["image_center"], dtype=float),
        "c2w": np.asarray(calib["polynomialC2W"], dtype=float),   # = -taylor
        "w2c": np.asarray(calib["polynomialW2C"], dtype=float),   # ascending, EgoRear angle
        "size": calib["size"],                                     # [H, W]
        "name": calib.get("name", path),
    }


def pixels_to_normalized(uv, intr):
    """Corner pixels (N,2) -> pinhole-normalized coords (N,2) via the exact
    fisheye ray model. Rays behind the image plane (z<=0, >=180 deg FOV
    points) are returned as nan — the caller must mask them."""
    p = uv - intr["center"]
    rho = np.linalg.norm(p, axis=1)
    # EgoRear stores C2W = -taylor; ray z toward the scene is +taylor(rho)
    z = -np.polynomial.polynomial.polyval(rho, intr["c2w"])
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(z[:, None] > 1e-6, p / z[:, None], np.nan)
    return m


def egorear_project(pts_cam, intr):
    """Camera-frame 3D points (N,3) -> pixels (N,2), EgoRear W2C formula."""
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    norm = np.sqrt(x * x + y * y)
    theta = np.arctan(-z / np.maximum(norm, 1e-12))
    rho = np.polynomial.polynomial.polyval(theta, intr["w2c"])
    u = x / norm * rho + intr["center"][0]
    v = y / norm * rho + intr["center"][1]
    return np.stack([u, v], axis=1)


def board_points_mm(pattern_size, square_mm):
    """Chessboard inner-corner grid in the board frame (N,3), millimeters,
    ordered the way OpenCV detectors emit corners (row-major)."""
    rows, cols = pattern_size
    g = np.zeros((rows * cols, 3))
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return g * square_mm


def solve_board_pose(corners_uv, board_pts, intr):
    """Board pose in the camera frame from one view. Returns 4x4 T (mm) with
    X_cam = T @ X_board, or None if PnP fails."""
    m = pixels_to_normalized(corners_uv, intr)
    mask = np.isfinite(m).all(axis=1)
    if mask.sum() < 6:
        return None
    obj = np.ascontiguousarray(board_pts[mask], dtype=np.float64)
    img = np.ascontiguousarray(m[mask], dtype=np.float64).reshape(-1, 1, 2)
    ok, rvecs, tvecs, err = cv2.solvePnPGeneric(
        obj, img, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok or len(rvecs) == 0:
        return None
    rvec, tvec = rvecs[int(np.argmin(err))], tvecs[int(np.argmin(err))]
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def relative_transform(T_left_board, T_right_board):
    """T mapping right-camera coords -> left-camera coords."""
    return T_left_board @ np.linalg.inv(T_right_board)


def average_transforms(T_list, rot_tol_deg=3.0, trans_tol_mm=15.0):
    """Robust average of rigid transforms: reject outliers vs the geometric
    median, then mean the survivors. Returns (T_avg, kept_mask, stats)."""
    T_list = np.asarray(T_list)
    quats = Rotation.from_matrix(T_list[:, :3, :3]).as_quat()
    quats[quats[:, 3] < 0] *= -1
    q_med = np.median(quats, axis=0)
    q_med /= np.linalg.norm(q_med)
    t_med = np.median(T_list[:, :3, 3], axis=0)

    R_med = Rotation.from_quat(q_med)
    rot_err = np.degrees([(R_med.inv() * Rotation.from_matrix(T[:3, :3])).magnitude()
                          for T in T_list])
    trans_err = np.linalg.norm(T_list[:, :3, 3] - t_med, axis=1)
    kept = (rot_err < rot_tol_deg) & (trans_err < trans_tol_mm)
    if kept.sum() < max(3, int(0.3 * len(T_list))):
        raise RuntimeError(
            f"no consensus among pairs: only {int(kept.sum())}/{len(T_list)} "
            f"agree with the median (rotation spread {np.median(rot_err):.1f} deg, "
            f"translation spread {np.median(trans_err):.0f} mm). This means the "
            f"per-pair poses are unreliable — usual causes: intrinsics from a "
            f"different resolution, wrong board rows/cols/square size, or the "
            f"board moving between the two grabs.")

    R_avg = Rotation.from_quat(quats[kept]).mean().as_matrix()
    t_avg = T_list[kept][:, :3, 3].mean(axis=0)
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg

    stats = {
        "n_pairs": int(len(T_list)),
        "n_kept": int(kept.sum()),
        "rot_spread_deg": float(np.std(rot_err[kept])),
        "trans_spread_mm": float(np.std(trans_err[kept])),
        "baseline_mm": float(np.linalg.norm(t_avg)),
    }
    return T_avg, kept, stats


def make_device_frame(T_left_from_right):
    """Symmetric device frame between the two cameras: origin at the midpoint
    of the camera centers, orientation halfway (slerp 0.5) between them.
    Returns (T_dev_to_left, T_dev_to_right): X_cam = T @ X_device, in the
    same length unit as the input transform."""
    # In left-camera coordinates: left camera at origin/identity; the right
    # camera center is at T_left_from_right[:3, 3] with rotation R_lr.
    R_lr = Rotation.from_matrix(T_left_from_right[:3, :3])
    c_right = T_left_from_right[:3, 3]

    R_dev = Rotation.concatenate([Rotation.identity(), R_lr]).mean()  # slerp 0.5
    c_dev = c_right / 2.0

    # Device pose in left-cam coords IS the device->left transform:
    # X_left = R_dev @ X_dev + c_dev.
    T_dev_to_left = np.eye(4)
    T_dev_to_left[:3, :3] = R_dev.as_matrix()
    T_dev_to_left[:3, 3] = c_dev

    # device->right = (right<-left) @ (left<-device)
    T_dev_to_right = np.linalg.inv(T_left_from_right) @ T_dev_to_left
    return T_dev_to_left, T_dev_to_right


def cross_reprojection_error(pairs_T_left, pairs_T_right, pairs_corners_right,
                             board_pts, T_left_from_right, intr_right):
    """Validation: board pose seen from LEFT, mapped through the calibrated
    extrinsic into the RIGHT camera, projected with RIGHT intrinsics, compared
    to the corners actually detected in the right image. Pixel RMS."""
    errs = []
    T_right_from_left = np.linalg.inv(T_left_from_right)
    for T_lb, uv_r in zip(pairs_T_left, pairs_corners_right):
        pts_left = (T_lb @ np.hstack([board_pts, np.ones((len(board_pts), 1))]).T).T[:, :3]
        pts_right = (T_right_from_left @ np.hstack(
            [pts_left, np.ones((len(pts_left), 1))]).T).T[:, :3]
        proj = egorear_project(pts_right, intr_right)
        errs.append(np.linalg.norm(proj - uv_r, axis=1))
    e = np.concatenate(errs)
    return float(np.sqrt((e ** 2).mean()))


def calibrate_extrinsics(pair_corners, pattern_size, square_mm,
                         intr_left, intr_right):
    """pair_corners: list of (corners_left, corners_right), each (N,2) pixels
    in the SAME corner order as emitted by the detector for that view.
    Returns (result_dict, report_lines) or raises RuntimeError."""
    board_pts = board_points_mm(pattern_size, square_mm)
    report = []

    T_lefts, T_rights, rel, corners_r = [], [], [], []
    for uv_l, uv_r in pair_corners:
        T_l = solve_board_pose(uv_l, board_pts, intr_left)
        T_r = solve_board_pose(uv_r, board_pts, intr_right)
        if T_l is None or T_r is None:
            continue
        T_lefts.append(T_l)
        T_rights.append(T_r)
        rel.append(relative_transform(T_l, T_r))
        corners_r.append(uv_r)
    if len(rel) < 3:
        raise RuntimeError(f"only {len(rel)} usable pairs — capture more views "
                           f"where BOTH cameras see the board")

    T_lr, kept, stats = average_transforms(rel)
    report.append(f"pairs used: {stats['n_kept']}/{stats['n_pairs']} "
                  f"(outliers = board-orientation flips or motion desync)")
    report.append(f"consistency: rotation spread {stats['rot_spread_deg']:.3f} deg, "
                  f"translation spread {stats['trans_spread_mm']:.2f} mm")
    report.append(f"baseline |t| = {stats['baseline_mm']:.1f} mm "
                  f"— check this against your physical rig!")

    rms = cross_reprojection_error(
        [T for T, k in zip(T_lefts, kept) if k],
        [T for T, k in zip(T_rights, kept) if k],
        [c for c, k in zip(corners_r, kept) if k],
        board_pts, T_lr, intr_right)
    report.append(f"cross-view reprojection RMS: {rms:.2f} px")

    T_dev_l_mm, T_dev_r_mm = make_device_frame(T_lr)
    mm2m = 0.001
    T_dev_l = T_dev_l_mm.copy()
    T_dev_r = T_dev_r_mm.copy()
    T_dev_l[:3, 3] *= mm2m
    T_dev_r[:3, 3] *= mm2m
    T_lr_m = T_lr.copy()
    T_lr_m[:3, 3] *= mm2m

    result = {
        "coord_transformation_matrix": {
            "device_to_camera_front_left": T_dev_l.tolist(),
            "device_to_camera_front_right": T_dev_r.tolist(),
        },
        "camera_right_to_left": T_lr_m.tolist(),
        "units": "meters",
        "device_frame": "midpoint of camera centers, orientation halfway "
                        "between the two cameras (OpenCV axes: x right, "
                        "y down, z forward)",
        "stats": stats | {"cross_reprojection_rms_px": rms},
        "intrinsics": {"left": intr_left["name"], "right": intr_right["name"]},
    }
    return result, report
