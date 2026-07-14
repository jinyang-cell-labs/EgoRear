# Fisheye intrinsic calibration GUI for EgoRear (one camera per session).
#
# Workflow: pick a /dev/video* device and resolution -> live preview with
# chessboard detection overlay -> capture frames (Space) -> Calibrate
# (py-OCamCalib) -> Save the EgoRear-format calibration JSON.
# All intermediate files (captured frames) live in a temp dir and are removed
# on exit; the only artifact the user sees is the final JSON.
#
# Launched by calibration_gui.bash, which prepares the py-OCamCalib venv.

import atexit
import glob
import os
import shutil
import sys
import tempfile
import threading

import matplotlib
matplotlib.use("Agg")  # py-OCamCalib calls plt.show(); never open windows

import cv2
import numpy as np
from loguru import logger

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyocamcalib_to_egorear import convert_to_egorear

COVERAGE_GRID = 4
RESOLUTIONS = ["640x480", "800x600", "1280x720", "1280x960", "1920x1080", "2560x1440"]


def list_video_devices():
    devices = []
    for dev in sorted(glob.glob("/dev/video*"), key=lambda d: int(d[10:])):
        name_file = f"/sys/class/video4linux/{os.path.basename(dev)}/name"
        try:
            with open(name_file) as f:
                name = f.read().strip()
        except OSError:
            name = "?"
        devices.append((dev, name))
    return devices


class CaptureThread(QThread):
    frame_ready = Signal(np.ndarray, bool)  # preview frame (BGR), board hint
    error = Signal(str)

    def __init__(self, device, width, height, pattern_size):
        super().__init__()
        self.device = device
        self.width = width
        self.height = height
        self.pattern_size = pattern_size  # (rows, cols) inner corners
        self._running = True
        self._lock = threading.Lock()
        self._latest = None
        self.actual_size = None

    def latest_frame(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self):
        self._running = False
        self.wait(2000)

    def run(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            self.error.emit(f"could not open {self.device}")
            return
        self.actual_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        n = 0
        hint = False
        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.error.emit("frame grab failed")
                break
            with self._lock:
                self._latest = frame
            n += 1
            if n % 5 == 0:  # cheap detection hint on a downscaled frame
                small = cv2.resize(frame, None, fx=520 / frame.shape[0],
                                   fy=520 / frame.shape[0])
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hint, _ = cv2.findChessboardCorners(
                    gray, self.pattern_size,
                    flags=cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH)
            self.frame_ready.emit(frame, bool(hint))
        cap.release()


class CalibrationThread(QThread):
    log = Signal(str)
    done = Signal(object, object, str)  # egorear_dict, report_lines, error

    def __init__(self, work_dir, pattern_size, square_size, cam_name, frame_size):
        super().__init__()
        self.work_dir = work_dir
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.cam_name = cam_name
        self.frame_size = frame_size  # (w, h)

    def run(self):
        sink_id = logger.add(lambda msg: self.log.emit(msg.strip()),
                             format="{message}", level="INFO")
        try:
            from pyocamcalib.modelling.calibration import CalibrationEngine
            engine = CalibrationEngine(self.work_dir, self.pattern_size,
                                       self.cam_name, self.square_size)
            engine.detect_corners(check=False)
            if len(engine.detections) < 5:
                raise RuntimeError(
                    f"only {len(engine.detections)} images had a detected board "
                    f"— capture more views or check the rows/cols values")
            engine.estimate_fisheye_parameters()
            engine.find_poly_inv()
            out, report = convert_to_egorear(
                engine.taylor_coefficient, engine.inverse_poly,
                engine.distortion_center, engine.stretch_matrix,
                self.frame_size[0], self.frame_size[1], self.cam_name)
            report.insert(0, f"overall reprojection RMS: {engine.rms_overall:.3f} px "
                             f"({len(engine.detections)} images used)")
            self.done.emit(out, report, "")
        except Exception as e:  # surfaced in the GUI, not a crash
            self.done.emit(None, None, str(e))
        finally:
            logger.remove(sink_id)


class CoverageWidget(QLabel):
    """Shows which image regions already have board samples (green cells)."""

    def __init__(self):
        super().__init__()
        self.covered = set()
        self.setFixedSize(120, 90)
        self._render()

    def add_corners(self, corners, frame_size):
        w, h = frame_size
        for x, y in corners:
            cx = min(int(x / w * COVERAGE_GRID), COVERAGE_GRID - 1)
            cy = min(int(y / h * COVERAGE_GRID), COVERAGE_GRID - 1)
            self.covered.add((cx, cy))
        self._render()

    def reset(self):
        self.covered = set()
        self._render()

    def _render(self):
        img = np.full((90, 120, 3), 40, np.uint8)
        cw, ch = 120 // COVERAGE_GRID, 90 // COVERAGE_GRID
        for gy in range(COVERAGE_GRID):
            for gx in range(COVERAGE_GRID):
                color = (60, 160, 60) if (gx, gy) in self.covered else (60, 60, 60)
                cv2.rectangle(img, (gx * cw + 1, gy * ch + 1),
                              ((gx + 1) * cw - 2, (gy + 1) * ch - 2), color, -1)
        qimg = QImage(img.data, 120, 90, 360, QImage.Format_BGR888)
        self.setPixmap(QPixmap.fromImage(qimg))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EgoRear fisheye intrinsic calibration")
        self.capture_thread = None
        self.calib_thread = None
        self.calib_result = None
        self.n_captured = 0
        self.saved_files = []
        self.board_hint = False
        self.work_dir = tempfile.mkdtemp(prefix="egorear_calib_")
        atexit.register(shutil.rmtree, self.work_dir, ignore_errors=True)
        self._build_ui()
        self._refresh_devices()

    # ---------- UI ----------
    def _build_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)

        panel = QVBoxLayout()

        cam_box = QGroupBox("Camera")
        g = QGridLayout(cam_box)
        self.device_combo = QComboBox()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_devices)
        self.res_combo = QComboBox()
        self.res_combo.setEditable(True)
        self.res_combo.addItems(RESOLUTIONS)
        self.res_combo.setCurrentText("1280x720")
        self.open_btn = QPushButton("Open camera")
        self.open_btn.clicked.connect(self._toggle_camera)
        self.cam_status = QLabel("closed")
        g.addWidget(QLabel("Device"), 0, 0)
        g.addWidget(self.device_combo, 0, 1)
        g.addWidget(refresh_btn, 0, 2)
        g.addWidget(QLabel("Resolution"), 1, 0)
        g.addWidget(self.res_combo, 1, 1, 1, 2)
        g.addWidget(self.open_btn, 2, 0, 1, 2)
        g.addWidget(self.cam_status, 2, 2)
        panel.addWidget(cam_box)

        board_box = QGroupBox("Chessboard (INNER corners, not squares)")
        g = QGridLayout(board_box)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 30)
        self.rows_spin.setValue(6)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 30)
        self.cols_spin.setValue(8)
        self.square_spin = QDoubleSpinBox()
        self.square_spin.setRange(1.0, 500.0)
        self.square_spin.setValue(30.0)
        self.square_spin.setSuffix(" mm")
        g.addWidget(QLabel("Rows"), 0, 0)
        g.addWidget(self.rows_spin, 0, 1)
        g.addWidget(QLabel("Columns"), 0, 2)
        g.addWidget(self.cols_spin, 0, 3)
        g.addWidget(QLabel("Square size"), 1, 0)
        g.addWidget(self.square_spin, 1, 1, 1, 3)
        g.addWidget(QLabel("If detection never triggers, swap rows/columns."), 2, 0, 1, 4)
        panel.addWidget(board_box)

        cap_box = QGroupBox("Capture")
        g = QGridLayout(cap_box)
        self.capture_btn = QPushButton("Capture  (Space)")
        self.capture_btn.clicked.connect(self._capture)
        self.capture_btn.setEnabled(False)
        self.undo_btn = QPushButton("Discard last")
        self.undo_btn.clicked.connect(self._undo)
        self.undo_btn.setEnabled(False)
        self.count_label = QLabel("0 captured (aim for 30–60)")
        self.coverage = CoverageWidget()
        g.addWidget(self.capture_btn, 0, 0)
        g.addWidget(self.undo_btn, 0, 1)
        g.addWidget(self.count_label, 1, 0, 1, 2)
        g.addWidget(QLabel("Coverage"), 2, 0)
        g.addWidget(self.coverage, 2, 1)
        panel.addWidget(cap_box)

        calib_box = QGroupBox("Calibrate && save")
        g = QGridLayout(calib_box)
        self.name_combo = QComboBox()
        self.name_combo.setEditable(True)
        self.name_combo.addItems(["camera_front_left", "camera_front_right",
                                  "camera_back_left", "camera_back_right"])
        self.calib_btn = QPushButton("Calibrate")
        self.calib_btn.clicked.connect(self._calibrate)
        self.calib_btn.setEnabled(False)
        self.save_btn = QPushButton("Save EgoRear JSON…")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        g.addWidget(QLabel("Camera name"), 0, 0)
        g.addWidget(self.name_combo, 0, 1)
        g.addWidget(self.calib_btn, 1, 0)
        g.addWidget(self.save_btn, 1, 1)
        panel.addWidget(calib_box)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        panel.addWidget(self.log_box, stretch=1)

        left = QWidget()
        left.setLayout(panel)
        left.setFixedWidth(380)
        layout.addWidget(left)

        self.preview = QLabel("open a camera to start")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.preview.setStyleSheet("background: #202020; color: #808080;")
        layout.addWidget(self.preview, stretch=1)

        self.setCentralWidget(root)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._capture)

    def _log(self, msg):
        self.log_box.appendPlainText(msg)

    # ---------- camera ----------
    def _refresh_devices(self):
        self.device_combo.clear()
        for dev, name in list_video_devices():
            self.device_combo.addItem(f"{dev}  ({name})", dev)
        if self.device_combo.count() == 0:
            self.device_combo.addItem("no /dev/video* found", None)

    def _pattern_size(self):
        return (self.rows_spin.value(), self.cols_spin.value())

    def _toggle_camera(self):
        if self.capture_thread is not None:
            self.capture_thread.stop()
            self.capture_thread = None
            self.open_btn.setText("Open camera")
            self.cam_status.setText("closed")
            self.capture_btn.setEnabled(False)
            return
        dev = self.device_combo.currentData()
        if not dev:
            QMessageBox.warning(self, "No device", "No video device selected.")
            return
        try:
            w, h = map(int, self.res_combo.currentText().lower().split("x"))
        except ValueError:
            QMessageBox.warning(self, "Bad resolution", "Use e.g. 1280x720.")
            return
        self.capture_thread = CaptureThread(dev, w, h, self._pattern_size())
        self.capture_thread.frame_ready.connect(self._show_frame)
        self.capture_thread.error.connect(self._camera_error)
        self.capture_thread.start()
        self.open_btn.setText("Close camera")
        self.capture_btn.setEnabled(True)
        QTimer.singleShot(1500, self._report_actual_size)

    def _report_actual_size(self):
        if self.capture_thread and self.capture_thread.actual_size:
            w, h = self.capture_thread.actual_size
            self.cam_status.setText(f"{w}x{h}")
            self._log(f"camera delivers {w}x{h} "
                      f"(calibration will be valid for this resolution only)")

    def _camera_error(self, msg):
        self._log(f"camera error: {msg}")
        self._toggle_camera()

    def _show_frame(self, frame, board_hint):
        self.board_hint = board_hint
        h, w = frame.shape[:2]
        border = (0, 200, 0) if board_hint else (0, 0, 200)
        frame = cv2.copyMakeBorder(frame, 6, 6, 6, 6, cv2.BORDER_CONSTANT,
                                   value=border)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3,
                      QImage.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ---------- capture ----------
    def _capture(self):
        if self.capture_thread is None:
            return
        frame = self.capture_thread.latest_frame()
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = 520.0 / gray.shape[0]
        small = cv2.resize(gray, None, fx=scale, fy=scale)
        ok, corners = cv2.findChessboardCornersSB(
            small, self._pattern_size(), flags=cv2.CALIB_CB_EXHAUSTIVE)
        if not ok:
            self._log("rejected: no chessboard detected in this frame")
            return
        corners = corners.reshape(-1, 2) / scale
        path = os.path.join(self.work_dir, f"frame_{self.n_captured:04d}.png")
        cv2.imwrite(path, frame)
        self.saved_files.append((path, corners))
        self.n_captured += 1
        self.coverage.add_corners(corners, (frame.shape[1], frame.shape[0]))
        self.count_label.setText(f"{self.n_captured} captured (aim for 30–60)")
        self.undo_btn.setEnabled(True)
        self.calib_btn.setEnabled(self.n_captured >= 5)
        self._log(f"captured frame {self.n_captured}")

    def _undo(self):
        if not self.saved_files:
            return
        path, _ = self.saved_files.pop()
        os.remove(path)
        self.n_captured -= 1
        self.count_label.setText(f"{self.n_captured} captured (aim for 30–60)")
        self.coverage.reset()
        for p, corners in self.saved_files:
            img = cv2.imread(p)
            self.coverage.add_corners(corners, (img.shape[1], img.shape[0]))
        self.undo_btn.setEnabled(bool(self.saved_files))
        self.calib_btn.setEnabled(self.n_captured >= 5)
        self._log("discarded last capture")

    # ---------- calibration ----------
    def _calibrate(self):
        if not self.saved_files:
            return
        img = cv2.imread(self.saved_files[0][0])
        frame_size = (img.shape[1], img.shape[0])
        self.calib_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self._log("=== calibration started (this can take a few minutes) ===")
        self.calib_thread = CalibrationThread(
            self.work_dir, self._pattern_size(), self.square_spin.value(),
            self.name_combo.currentText().strip(), frame_size)
        self.calib_thread.log.connect(self._log)
        self.calib_thread.done.connect(self._calib_done)
        self.calib_thread.start()

    def _calib_done(self, result, report, error):
        self.calib_btn.setEnabled(True)
        if error:
            self._log(f"calibration FAILED: {error}")
            QMessageBox.critical(self, "Calibration failed", error)
            return
        self.calib_result = result
        for line in report:
            self._log(line)
        self._log("=== calibration finished — review RMS above, then Save ===")
        self.save_btn.setEnabled(True)

    def _save(self):
        if self.calib_result is None:
            return
        name = self.name_combo.currentText().strip()
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "my_rig", f"{name}.json")
        os.makedirs(os.path.dirname(default), exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save EgoRear calibration", default, "JSON (*.json)")
        if not path:
            return
        import json
        self.calib_result["name"] = f"{name}_scaramuzza"
        with open(path, "w") as f:
            json.dump(self.calib_result, f, indent=2)
        self._log(f"saved {path}")

    def closeEvent(self, event):
        if self.capture_thread:
            self.capture_thread.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(1200, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
