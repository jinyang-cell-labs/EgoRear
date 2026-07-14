# Fast drop-in replacement for py-OCamCalib's bundle adjustment.
#
# The upstream implementation (pyocamcalib.core.optim.bundle_adjustement) is
# correct but slow for two structural reasons:
#   1. The residual solves a quartic with np.roots PER CORNER POINT in Python
#      loops (~3000 solves per evaluation for 25 images).
#   2. least_squares(method='lm') builds a DENSE finite-difference Jacobian:
#      one residual evaluation per parameter (12*N_images + 9), even though
#      image i's pose only affects image i's residuals.
#
# This module keeps the exact same cost function (bit-identical projection
# math, including upstream's stretch-matrix application order) but:
#   - batches all quartic solves into one vectorized companion-matrix
#     eigenvalue call, and
#   - uses method='trf' with a block-sparse jac_sparsity, so a Jacobian
#     costs ~20 grouped evaluations instead of ~300+.
#
# Apply with install() BEFORE constructing a CalibrationEngine.

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _batched_min_positive_real_root(a4, a3, a2, a1, a0):
    """Smallest positive real root of a4 x^4 + a3 x^3 + a2 x^2 + a1 x + a0
    for each element of the input arrays (all shape (P,)). Returns (P,) with
    np.nan where no positive real root exists.

    Same result as np.roots per point (companion-matrix eigenvalues), but one
    batched eigvals call instead of P Python-level calls.
    """
    P = a1.shape[0]
    comp = np.zeros((P, 4, 4))
    comp[:, 1, 0] = 1.0
    comp[:, 2, 1] = 1.0
    comp[:, 3, 2] = 1.0
    comp[:, 0, 3] = -a0 / a4
    comp[:, 1, 3] = -a1 / a4
    comp[:, 2, 3] = -a2 / a4
    comp[:, 3, 3] = -a3 / a4
    roots = np.linalg.eigvals(comp)  # (P, 4) complex
    positive_real = (np.abs(roots.imag) < 1e-10) & (roots.real > 0)
    real = np.where(positive_real, roots.real, np.inf)
    rho = real.min(axis=1)
    rho[~np.isfinite(rho)] = np.nan
    return rho


def _project(world_points_cam, taylor, distortion_center, stretch_matrix):
    """Vectorized replica of pyocamcalib Camera.world2cam (incl. its stretch
    application order) for points already in the camera frame, (P, 3) -> (P, 2)."""
    pts = world_points_cam.copy()
    pts[:, :2][pts[:, :2] == 0] = np.finfo(float).eps
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    z_scaled = pts[:, 2] / r

    # taylor = (a0, a1=0, a2, a3, a4); solve a4 r^4+a3 r^3+a2 r^2+(a1-z)r+a0=0
    a0, a1, a2, a3, a4 = taylor
    rho = _batched_min_positive_real_root(
        np.full_like(z_scaled, a4), np.full_like(z_scaled, a3),
        np.full_like(z_scaled, a2), a1 - z_scaled,
        np.full_like(z_scaled, a0))

    x = (pts[:, 0] / r) * rho
    y = (pts[:, 1] / r) * rho
    # NOTE: replicates upstream exactly — y uses the already-stretched x.
    x = x * stretch_matrix[0, 0] + y * stretch_matrix[0, 1]
    y = x * stretch_matrix[1, 0] + y
    return np.stack([x + distortion_center[0], y + distortion_center[1]], axis=1)


def _unpack(x, N):
    extrinsics_t = [np.array(x[i * 12:(i + 1) * 12]).reshape(3, 4) for i in range(N)]
    stretch_matrix = np.array([[x[12 * N], x[12 * N + 1]],
                               [x[12 * N + 2], 1]])
    distortion_center = (x[12 * N + 3], x[12 * N + 4])
    taylor_coefficient = np.array([x[12 * N + 5], 0, *x[12 * N + 6:]])
    return extrinsics_t, stretch_matrix, distortion_center, taylor_coefficient


def fast_bundle_adjustement(data, valid, extrinsics_t_init,
                            distortion_center_init, taylor_coefficient_init):
    """Same signature/return as pyocamcalib.core.optim.bundle_adjustement."""
    # Gather per-image points once, homogeneous world points for 3x4 extrinsics.
    imgs = []
    for idx, img_path in enumerate(sorted(data.keys())):
        if valid[idx]:
            ip = np.asarray(data[img_path]["image_points"], dtype=float)
            wp = np.asarray(data[img_path]["world_points"], dtype=float)
            wp_h = np.hstack([wp, np.ones((wp.shape[0], 1))])
            imgs.append((ip, wp_h))
    N = len(imgs)
    counts = [ip.shape[0] for ip, _ in imgs]

    x0 = []
    for i in range(N):
        x0.extend(list(np.asarray(extrinsics_t_init[i]).flatten()))
    x0.extend([1, 0, 0])
    x0.extend(list(distortion_center_init))
    ss = [taylor_coefficient_init[0], *taylor_coefficient_init[2:]]
    x0.extend(ss)
    x0 = np.array(x0, dtype=float)

    n_params = x0.shape[0]
    n_global = n_params - 12 * N

    def residuals(x):
        extrinsics_t, stretch, center, taylor = _unpack(x, N)
        errs = []
        for i, (ip, wp_h) in enumerate(imgs):
            cam_pts = wp_h @ extrinsics_t[i].T  # (P, 3)
            proj = _project(cam_pts, taylor, center, stretch)
            err = ip - proj
            # upstream orders residuals [err_x..., err_y...] per image
            errs.append(np.concatenate([err[:, 0], err[:, 1]]))
        out = np.concatenate(errs)
        return np.nan_to_num(out, nan=1e3)  # unprojectable point -> big error

    # Block sparsity: image i's 12 pose params only touch its own residuals;
    # the 9 global params (stretch, center, taylor) touch everything.
    sparsity = lil_matrix((2 * sum(counts), n_params), dtype=np.uint8)
    row = 0
    for i, c in enumerate(counts):
        sparsity[row:row + 2 * c, 12 * i:12 * (i + 1)] = 1
        row += 2 * c
    sparsity[:, 12 * N:] = 1

    # Tolerance 1e-5 (tighter than upstream's 1e-4): with trf + x_scale='jac'
    # 1e-4 stops ~10% short of the converged RMS, while below 1e-5 the RMS no
    # longer improves and iterations are wasted (measured on the sample sets).
    result = least_squares(
        residuals, x0=x0, method="trf",
        jac_sparsity=sparsity, x_scale="jac",
        ftol=1e-5, xtol=1e-5, gtol=1e-5, max_nfev=300,
    )
    assert n_global == 9

    return _unpack(result.x, N)


def install():
    """Monkeypatch py-OCamCalib so CalibrationEngine uses the fast version."""
    import pyocamcalib.core.optim as optim
    import pyocamcalib.modelling.calibration as calibration
    optim.bundle_adjustement = fast_bundle_adjustement
    calibration.bundle_adjustement = fast_bundle_adjustement  # from-import binding
