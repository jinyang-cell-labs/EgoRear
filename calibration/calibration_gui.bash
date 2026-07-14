#!/usr/bin/env bash
# Launch the EgoRear fisheye intrinsic calibration GUI.
# Auto-provisions everything on first run: clones py-OCamCalib if missing,
# creates its venv, installs dependencies (incl. PySide6 for the GUI).
# Override the py-OCamCalib location with: PYOCAMCALIB_DIR=/path ./calibration_gui.bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYOCAM_DIR="${PYOCAMCALIB_DIR:-$HOME/repo/py-OCamCalib}"
VENV="$PYOCAM_DIR/.venv"

if [ ! -d "$PYOCAM_DIR" ]; then
    echo "[setup] cloning py-OCamCalib into $PYOCAM_DIR ..."
    git clone --depth 1 https://github.com/jakarto3d/py-OCamCalib.git "$PYOCAM_DIR"
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "[setup] creating venv and installing py-OCamCalib (one-time, ~2 min) ..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
    "$VENV/bin/pip" install --quiet -r "$PYOCAM_DIR/requirements.txt"
    "$VENV/bin/pip" install --quiet -e "$PYOCAM_DIR"
fi

if ! "$VENV/bin/python" -c "import PySide6" 2>/dev/null; then
    echo "[setup] installing PySide6 (GUI toolkit) ..."
    "$VENV/bin/pip" install --quiet PySide6
fi

# opencv-python (full) bundles its own Qt, which clashes with PySide6 in one
# process; swap it for the headless build (identical cv2 API, no GUI code).
if "$VENV/bin/pip" show --quiet opencv-python 2>/dev/null; then
    echo "[setup] replacing opencv-python with opencv-python-headless (Qt conflict) ..."
    "$VENV/bin/pip" uninstall --quiet -y opencv-python
    "$VENV/bin/pip" install --quiet opencv-python-headless
fi

# Qt platform selection: PySide6's X11 (xcb) plugin needs libxcb-cursor0,
# which Ubuntu doesn't ship by default. Prefer Wayland when available,
# otherwise verify the library exists and fail with a clear message.
if [ -z "${QT_QPA_PLATFORM:-}" ]; then
    if [ -n "${WAYLAND_DISPLAY:-}" ]; then
        export QT_QPA_PLATFORM=wayland
    elif ! ldconfig -p | grep -q libxcb-cursor.so \
         && ! ls /usr/lib/*/libxcb-cursor.so.0 >/dev/null 2>&1; then
        echo "ERROR: your session is X11 and libxcb-cursor0 is not installed."
        echo "       Qt 6 needs it to open a window. Install it once with:"
        echo ""
        echo "           sudo apt install libxcb-cursor0"
        echo ""
        echo "       then re-run this script."
        exit 1
    fi
fi

# MPLBACKEND=Agg: py-OCamCalib calls plt.show() internally; never pop windows.
exec env MPLBACKEND=Agg "$VENV/bin/python" "$SCRIPT_DIR/calibration_gui.py" "$@"
