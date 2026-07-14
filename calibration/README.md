# Camera Calibration Tools for EgoRear

Tools to calibrate your own fisheye cameras into the Scaramuzza calibration
format that EgoRear expects (see
`pose_estimation/utils/camera_calib_file/ego4view/*.json`), so you can run the
pretrained models on your own rig.

Intrinsic calibration is powered by
[py-OCamCalib](https://github.com/jakarto3d/py-OCamCalib), a Python
implementation of Scaramuzza's OCamCalib toolbox.

## Contents

| File | Purpose |
|---|---|
| `calibration_gui.bash` | One-command launcher: provisions everything, then starts the GUI |
| `calibration_gui.py` | GUI: select camera → capture chessboard views → calibrate → save EgoRear JSON |
| `pyocamcalib_to_egorear.py` | CLI converter: py-OCamCalib calibration JSON → EgoRear calibration JSON |

## Quick start (GUI)

```bash
./calibration/calibration_gui.bash
```

First run auto-provisions everything (no sudo needed):

- clones py-OCamCalib into `~/repo/py-OCamCalib` (override: `PYOCAMCALIB_DIR=/path ./calibration_gui.bash`)
- creates its venv and installs dependencies, incl. PySide6 for the GUI

Later runs start the GUI directly.

### Workflow (run once per camera)

1. **Camera** — pick your device from the list (`/dev/video*` with names),
   set the resolution, click *Open camera*. The status shows the resolution
   the camera actually delivers — **the calibration is only valid at that
   resolution**, so use the same one at runtime.
2. **Chessboard** — enter rows/columns of **inner corners** (a board with
   9×7 squares has 8×6 inner corners) and the measured square size in mm.
3. **Capture** — the preview border turns **green** when the board is
   detected. Press **Space** (or *Capture*) to grab a frame; frames without a
   clean detection are rejected. Or enable **Auto-capture** to grab a frame
   automatically at a set rate (default 1 Hz) whenever the border is green —
   then just move the board slowly through the whole field of view. Aim for
   **~25 captures** and fill the whole coverage map, especially the image
   edges — that is where fisheye calibrations go wrong. Vary the board's
   distance and tilt. (More captures don't hurt: the optimization runtime
   grows steeply with image count, so calibration automatically uses at most
   25 evenly-spaced captures.)
4. **Calibrate** — takes a few minutes; progress streams into the log. Check
   the reported numbers: overall reprojection **RMS should be well under
   1 px**; the refit/cross-check lines should be sub-0.1 px.
5. **Save EgoRear JSON…** — defaults to `calibration/my_rig/<camera_name>.json`.
   This is the only file produced; captured frames live in a temp dir and are
   deleted on exit.

For a stereo pair, run the session twice: save as `camera_front_left.json`
and `camera_front_right.json`, then point the model config at the folder:

```yaml
camera_calib_file_dir_path: ./calibration/my_rig
```

## Extrinsic calibration (second GUI tab)

After calibrating both cameras' intrinsics, switch to the **Extrinsic (stereo
pair)** tab to calibrate the rigid transform between them and generate
EgoRear's `device_to_camera_*` matrices:

1. Select the **two** camera devices (metadata-only nodes are filtered out
   automatically) and the resolution, point the two file fields at the saved
   intrinsic JSONs, set the board parameters, *Open cameras*.
2. Both streams are shown **raw (distorted)** with a per-camera green/red
   detection border — no undistortion is needed anywhere; the math converts
   detected corners to exact 3D rays via the fisheye model and runs planar
   PnP on ray-normalized coordinates (identity camera matrix, no pinhole
   approximation).
3. Capture **10–20 pairs** (Space or auto-capture — requires both views
   green). **Hold the board still at each capture**: the two cameras are not
   hardware-synchronized, so board motion between the two grabs corrupts a
   pair. Vary board position/angle across the shared field of view.
4. *Calibrate extrinsics* — per-pair relative poses are averaged with
   median-based outlier rejection (this also removes pairs where the
   chessboard's 180° orientation ambiguity flipped the corner order in one
   view). Review the report:
   - **baseline** — compare against a ruler measurement of your rig!
   - **rotation/translation spread** — pair-to-pair consistency
   - **cross-view reprojection RMS** — should be a few pixels or less
5. *Save extrinsics JSON* — contains `coord_transformation_matrix.
   device_to_camera_front_left/right` (4×4, **meters**, ready for EgoRear's
   `coord_trans_mat` input) plus the raw left↔right transform and quality
   stats. The device frame is the midpoint between the cameras with the
   average of their orientations (OpenCV axes: x right, y down, z forward).

Validated on synthetic ground truth (real lens polynomials, 0.1 px corner
noise, deliberately flipped pairs): recovers the true transform to
0.05° / 0.5 mm and rejects the corrupted pairs.

## CLI converter (no GUI)

If you already calibrated with py-OCamCalib (or MATLAB OCamCalib exported to
its JSON format), convert the result directly:

```bash
python calibration/pyocamcalib_to_egorear.py \
    path/to/calibration_mycam_<date>.json \
    --width 1280 --height 960 --name camera_front_left -o ./calibration/my_rig
```

`--width/--height` must be the resolution the calibration images were taken
at. The script prints two self-checks (polynomial refit error and a
projection cross-check between both formulas); both should be sub-pixel.

## Why a converter is needed at all

py-OCamCalib and EgoRear both use the Scaramuzza camera model, but store it
differently — the coefficients cannot simply be copied:

- **`polynomialW2C`** (world→camera): py-OCamCalib stores coefficients
  highest-degree-first (`np.polyval` convention) as a function of the angle
  from the **optical axis**; EgoRear stores them lowest-degree-first as a
  function of the angle from the **image plane** (shifted by 90°, see
  `theta = atan(-z/norm)` in `pose_estimation/utils/camera_models.py`). The
  converter re-fits the polynomial in the shifted variable.
- **`polynomialC2W`** (camera→world): EgoRear keeps the original MATLAB
  OCamCalib sign convention (negative toward the scene) → coefficients are
  negated. Unused at inference, stored for completeness.
- **`stretch_matrix`** (sensor/lens misalignment): EgoRear's math has no
  counterpart (`affine` exists in the JSON but is never read by the code).
  The converter warns if your lens's stretch is non-identity enough to
  matter.

Verified against EgoRear's actual `FishEyeCameraCalibratedModel` (torch):
projections from a converted calibration agree with py-OCamCalib's own
`world2cam_fast` to < 1 px (mean ~0.3 px, dominated by the ignored stretch
matrix on the test lens).

## Coordinate conventions (for the extrinsics step)

EgoRear's camera frame is standard **OpenCV convention: x right, y down,
z forward** (verified against the released ego4view calibration values). The
per-frame `coord_trans_mat` input is the rigid 4×4 **device→camera** transform
in **meters**; 3D joint positions are in **centimeters** in the device frame
(the code converts internally). For a rigid camera rig these matrices are
constant — calibrate them once with synchronized views of a board from both
cameras.

## Troubleshooting

- **GUI window doesn't appear** — the launcher prefers Wayland; on X11
  sessions Qt needs `libxcb-cursor0`: `sudo apt install libxcb-cursor0`.
- **Border never turns green** with the board clearly in view — swap the
  rows/columns values (the most common mistake), check the counts are inner
  corners, improve lighting, hold the board flatter.
- **Calibration fails with "only N images had a detected board"** — the
  full-quality detector is stricter than the live hint; capture more views,
  avoid motion blur, keep the board fully inside the frame.
- **High RMS (> 1 px)** — usually a bent board, wrong square size, or too few
  captures near the image edges.
- **`could not open /dev/videoX`** — device busy (close other apps using the
  camera) or it's a metadata node; USB cameras typically expose two nodes,
  use the first of the pair.
