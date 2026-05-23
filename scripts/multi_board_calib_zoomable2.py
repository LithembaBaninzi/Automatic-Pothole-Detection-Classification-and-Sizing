"""
multi_board_calib_zoomable.py
------------------------------
Manually select the four outer corners of multiple checkerboards in one image,
with a zoomable/panable view for precise clicking.

Outputs two sets of scale tables:
  - _lr.npy  : linear regression (global fit)
  - _ie.npy  : interpolation with extrapolation (passes through all anchor points)
"""

import cv2
import numpy as np
import os
from scipy.interpolate import interp1d
from scipy.stats import linregress

script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Load intrinsics (for undistortion) ---
K    = np.load(os.path.join(script_dir,"K_matlab1.npy"))
dist = np.load(os.path.join(script_dir, "dist_matlab1.npy"))

# --- Constants ---
REAL_WIDTH_CM  = 29.6   # 8 squares × 28 mm
REAL_HEIGHT_CM = 21   # 7 squares × 28 mm
IMAGE_H = 1080
IMAGE_W = 1920

# --- Image path ---
IMG_PATH = os.path.join(script_dir, "calibration_images_17", "calib_01.jpg")

# --- Helper: load, rotate, undistort ---
def load_undistorted(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open {img_path}")
    img = cv2.rotate(img, cv2.ROTATE_180)
    return cv2.undistort(img, K, dist)

# ── Zoomable display (unchanged) ──────────────────────────────────────────────
class ZoomableImage:
    def __init__(self, img, window_name="Select corners"):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.window_name = window_name
        self.zoom = 1.0
        self.offset_x = self.w // 2
        self.offset_y = self.h // 2
        self.points = []
        self.last_crop = (0, 0, self.w, self.h, self.w, self.h)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        cv2.createTrackbar("Zoom x10", window_name, 10, 40, self.on_zoom_change)
        cv2.createTrackbar("Pan X",    window_name, self.w // 2, self.w, self.on_pan_x)
        cv2.createTrackbar("Pan Y",    window_name, self.h // 2, self.h, self.on_pan_y)

        self.update_display()

    def on_zoom_change(self, val):
        self.zoom = max(0.1, val / 10.0)
        self.update_display()

    def on_pan_x(self, val):
        self.offset_x = val
        self.update_display()

    def on_pan_y(self, val):
        self.offset_y = val
        self.update_display()

    def update_display(self):
        crop_w = max(1, int(self.w / self.zoom))
        crop_h = max(1, int(self.h / self.zoom))
        x1 = max(0, self.offset_x - crop_w // 2)
        y1 = max(0, self.offset_y - crop_h // 2)
        x2 = min(self.w, x1 + crop_w)
        y2 = min(self.h, y1 + crop_h)
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        crop = self.img[y1:y2, x1:x2]
        if crop.size == 0:
            return
        disp_w = max(1, int((x2 - x1) * self.zoom))
        disp_h = max(1, int((y2 - y1) * self.zoom))
        display = cv2.resize(crop, (disp_w, disp_h))

        for idx, pt in enumerate(self.points):
            dx = int((pt[0] - x1) * self.zoom)
            dy = int((pt[1] - y1) * self.zoom)
            if 0 <= dx < display.shape[1] and 0 <= dy < display.shape[0]:
                cv2.circle(display, (dx, dy), 6, (0, 0, 255), -1)
                cv2.putText(display, str(idx + 1), (dx + 8, dy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(display,
                    f"Zoom: {self.zoom:.1f}x  |  Points: {len(self.points)}/4  "
                    f"|  ENTER=confirm  ESC=skip",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow(self.window_name, display)
        self.last_crop = (x1, y1, x2, y2, crop_w, crop_h)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            x1, y1, x2, y2, crop_w, crop_h = self.last_crop
            orig_x = x1 + int(x / self.zoom)
            orig_y = y1 + int(y / self.zoom)
            orig_x = max(0, min(self.w - 1, orig_x))
            orig_y = max(0, min(self.h - 1, orig_y))
            self.points.append((orig_x, orig_y))
            print(f"  Corner {len(self.points)} = ({orig_x}, {orig_y})")
            self.update_display()

    def close(self):
        cv2.destroyWindow(self.window_name)

def select_board_corners(img, board_name):
    print(f"\n--- Select board: {board_name} ---")
    print("  Zoom/pan with trackbars. Click TL → TR → BL → BR.")
    print("  ENTER when done, ESC to skip.\n")
    app = ZoomableImage(img, f"Select {board_name}")
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and len(app.points) == 4:
            break
        if key == 27:
            app.close()
            return None
        app.update_display()
    pts = app.points[:]
    app.close()
    return pts

def compute_board_info(points, distance_m):
    tl, tr, bl, br = points
    px_w  = (abs(tr[0] - tl[0]) + abs(br[0] - bl[0])) / 2.0
    px_h  = (abs(bl[1] - tl[1]) + abs(br[1] - tr[1])) / 2.0
    top_r = (tl[1] + tr[1]) / 2.0
    bot_r = (bl[1] + br[1]) / 2.0
    mid_r = (top_r + bot_r) / 2.0
    sw    = REAL_WIDTH_CM  / px_w
    sh    = REAL_HEIGHT_CM / px_h
    print(f"  {distance_m}m: mid_row={mid_r:.1f}  "
          f"px_W={px_w:.1f}  px_H={px_h:.1f}  "
          f"scale_w={sw:.4f} cm/px  scale_h={sh:.4f} cm/px")
    return {"mid_row": mid_r, "px_width": px_w, "px_height": px_h,
            "scale_w": sw, "scale_h": sh, "distance_m": distance_m}

def main():
    img = load_undistorted(IMG_PATH)
    print(f"Loaded: {IMG_PATH}  shape={img.shape}")

    board_defs = [
        ("1.0 m board", 1.0),
        ("1.5 m board", 1.5),
        ("2.0 m board", 2.0),
    ]

    boards = []
    for name, d in board_defs:
        corners = select_board_corners(img, name)
        if corners is None:
            print(f"  Skipped {name}")
            continue
        info = compute_board_info(corners, d)
        boards.append(info)

    if len(boards) < 2:
        print("Need at least 2 boards. Exiting.")
        return

    boards.sort(key=lambda b: b["mid_row"])
    rows     = np.array([b["mid_row"]  for b in boards])
    scales_w = np.array([b["scale_w"]  for b in boards])
    scales_h = np.array([b["scale_h"]  for b in boards])

    print("\n--- Calibration data (sorted by row) ---")
    for r, sw, sh, b in zip(rows, scales_w, scales_h, boards):
        print(f"  {b['distance_m']}m  row={r:.1f}  "
              f"scale_w={sw:.4f} cm/px  scale_h={sh:.4f} cm/px")

    # ──────────────────────────────────────────────────────────────────────────
    # METHOD 1: Linear regression (global fit)
    # ──────────────────────────────────────────────────────────────────────────
    slope_w, intcpt_w, r_w, _, _ = linregress(rows, scales_w)
    slope_h, intcpt_h, r_h, _, _ = linregress(rows, scales_h)

    print(f"\n--- Linear regression (global fit) ---")
    print(f"  Width  : scale_w = {slope_w:.8f} × row + {intcpt_w:.6f}  (R²={r_w**2:.4f})")
    print(f"  Height : scale_h = {slope_h:.8f} × row + {intcpt_h:.6f}  (R²={r_h**2:.4f})")

    all_rows = np.arange(IMAGE_H, dtype=float)
    lr_w = slope_w * all_rows + intcpt_w
    lr_h = slope_h * all_rows + intcpt_h
    lr_w[lr_w <= 0] = np.nan
    lr_h[lr_h <= 0] = np.nan

    # ──────────────────────────────────────────────────────────────────────────
    # METHOD 2: Piecewise linear interpolation with extrapolation
    # ──────────────────────────────────────────────────────────────────────────
    f_w = interp1d(rows, scales_w, kind='linear', fill_value='extrapolate')
    f_h = interp1d(rows, scales_h, kind='linear', fill_value='extrapolate')
    ie_w = f_w(all_rows)
    ie_h = f_h(all_rows)
    ie_w[ie_w <= 0] = np.nan
    ie_h[ie_h <= 0] = np.nan

    # ──────────────────────────────────────────────────────────────────────────
    # Validation and comparison
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Validation on anchor boards ---")
    print("  Distance |   Method   | Width error | Height error")
    for b in boards:
        r = b["mid_row"]
        # linear regression
        sw_lr = slope_w * r + intcpt_w
        sh_lr = slope_h * r + intcpt_h
        est_w_lr = b["px_width"] * sw_lr
        est_h_lr = b["px_height"] * sh_lr
        err_w_lr = abs(est_w_lr - REAL_WIDTH_CM) / REAL_WIDTH_CM * 100
        err_h_lr = abs(est_h_lr - REAL_HEIGHT_CM) / REAL_HEIGHT_CM * 100
        # interpolation+extrapolation
        sw_ie = f_w(r)
        sh_ie = f_h(r)
        est_w_ie = b["px_width"] * sw_ie
        est_h_ie = b["px_height"] * sh_ie
        err_w_ie = abs(est_w_ie - REAL_WIDTH_CM) / REAL_WIDTH_CM * 100
        err_h_ie = abs(est_h_ie - REAL_HEIGHT_CM) / REAL_HEIGHT_CM * 100
        print(f"  {b['distance_m']:.1f}m      |    LR     | {err_w_lr:6.2f}%        | {err_h_lr:6.2f}%")
        print(f"           |    IE     | {err_w_ie:6.2f}%        | {err_h_ie:6.2f}%")

    # ──────────────────────────────────────────────────────────────────────────
    # Save both sets of tables
    # ──────────────────────────────────────────────────────────────────────────
    out_lr_w = os.path.join(script_dir, "scale_table_w_lr_17_out.npy")
    out_lr_h = os.path.join(script_dir, "scale_table_h_lr_17_out.npy")
    out_ie_w = os.path.join(script_dir, "scale_table_w_ie_17_out.npy")
    out_ie_h = os.path.join(script_dir, "scale_table_h_ie_17_out.npy")
    np.save(out_lr_w, lr_w)
    np.save(out_lr_h, lr_h)
    np.save(out_ie_w, ie_w)
    np.save(out_ie_h, ie_h)

    print(f"\nSaved linear regression (LR) tables:")
    print(f"  {out_lr_w}")
    print(f"  {out_lr_h}")
    print(f"\nSaved interpolation+extrapolation (IE) tables:")
    print(f"  {out_ie_w}")
    print(f"  {out_ie_h}")

    # ── Optional plot ─────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, data_lr, data_ie, measured, row_vals, label, colour_lr, colour_ie in [
            (axes[0], lr_w, ie_w, scales_w, rows, "Width scale (cm/px)", "blue", "cyan"),
            (axes[1], lr_h, ie_h, scales_h, rows, "Height scale (cm/px)", "red", "orange"),
        ]:
            ax.plot(all_rows, data_lr, '--', color=colour_lr, linewidth=1.5, label="Linear regression")
            ax.plot(all_rows, data_ie, '-', color=colour_ie, linewidth=1.5, label="Interp+extrap")
            ax.scatter(row_vals, measured, color='black', s=80, zorder=5, label="Anchor points")
            for r, s, b in zip(row_vals, measured, boards):
                ax.annotate(f"{b['distance_m']}m", (r, s),
                            textcoords="offset points", xytext=(6, 4), fontsize=9)
            ax.set_xlabel("Image row")
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.4)
            ax.set_xlim(0, IMAGE_H)
        plt.suptitle("Scale table comparison: LR (global fit) vs IE (anchor-exact extrapolation)")
        plt.tight_layout()
        out_plot = os.path.join(script_dir, "scale_comparison_outside_17.png")
        plt.savefig(out_plot, dpi=120)
        print(f"Saved comparison plot: {out_plot}")
        plt.show()
    except ImportError:
        print("matplotlib not available — skipping plot")

if __name__ == "__main__":
    main()