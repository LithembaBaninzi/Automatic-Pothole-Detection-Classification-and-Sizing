"""
pothole_measure_video.py  —  IE + repair box + logging
=======================================================
Extends the still-image pipeline to support video input/output.
Supports:
  - Still image
  - Video file (.mp4, .avi, etc.)
  - Live camera (Raspberry Pi camera or USB webcam)

All detection results are logged to CSV (one row per group per frame).
Annotated video/images are saved alongside the CSV.

──────────────────────────────────────────────────────
USAGE EXAMPLES
──────────────────────────────────────────────────────

# Still image (same as before)
python pothole_measure_video.py --image Pot_grid_12.jpg

# Video file
python pothole_measure_video.py --video road_footage.mp4

# Video file, no display window (headless Pi)
python pothole_measure_video.py --video road_footage.mp4 --no_display

# Live Pi camera (upside-down, auto-flipped)
python pothole_measure_video.py --camera 0

# Live camera, save output video, custom device ID
python pothole_measure_video.py --camera 0 --save_video --device RPI-02

# Process every Nth frame (faster on Pi, e.g. every 3rd frame)
python pothole_measure_video.py --video road.mp4 --frame_skip 3

# Custom confidence threshold
python pothole_measure_video.py --video road.mp4 --conf 0.45

# Show help
python pothole_measure_video.py --help

──────────────────────────────────────────────────────
OUTPUT FILES (written next to the script)
──────────────────────────────────────────────────────
  pothole_detections.csv      ← all detections (appended each run)
  <input_name>_annotated.mp4  ← annotated output video
  <input_name>_annotated.jpg  ← annotated output image (still mode)
"""

import cv2
import numpy as np
import os
import math
import argparse
import csv
import sys
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))

today = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Parse CLI arguments ────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Pothole detection, sizing and repair-box logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--image",  metavar="PATH",
                     help="Path to a still image")
    src.add_argument("--video",  metavar="PATH",
                     help="Path to a video file")
    src.add_argument("--camera", metavar="INDEX", type=int,
                     help="Camera index (0 = default, use 0 for Pi cam)")

    p.add_argument("--model",  default="best_seg.pt",
                   help="YOLOv8 model file (default: best_seg.pt)")
    p.add_argument("--device", default="RPi_001",
                   help="Device ID written to CSV (default: RPi_001)")
    p.add_argument("--conf",   default=0.35, type=float,
                   help="Detection confidence threshold (default: 0.35)")
    p.add_argument("--frame_skip", default=1, type=int,
                   help="Process every Nth frame (default: 1 = every frame)")
    p.add_argument("--no_display", action="store_true",
                   help="Disable on-screen preview (use for headless Pi)")
    p.add_argument("--save_video", action="store_true",
                   help="Save annotated video output")
    p.add_argument("--no_flip", action="store_true",
                   help="Skip the 180° flip (camera is mounted normally)")
    p.add_argument("--csv",    default=f"pothole_detections_{today}.csv",
                   help="CSV output filename (default: pothole_detections_{today}.csv)")
    return p.parse_args()


args = parse_args()

# ── Load calibration ───────────────────────────────────────────────────────────
K       = np.load(os.path.join(script_dir, "K_matlab1.npy"))
dist    = np.load(os.path.join(script_dir, "dist_matlab1.npy"))
scale_w = np.load(os.path.join(script_dir, "scale_table_w_ie_outside.npy"))
scale_h = np.load(os.path.join(script_dir, "scale_table_h_ie_outside.npy"))

print(f"Intrinsics: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}")
print(f"Scale_w   : {np.nanmin(scale_w):.4f} - {np.nanmax(scale_w):.4f} cm/px")
print(f"Scale_h   : {np.nanmin(scale_h):.4f} - {np.nanmax(scale_h):.4f} cm/px")

# ── Load YOLO ──────────────────────────────────────────────────────────────────
model_path = os.path.join(script_dir, args.model)
model      = YOLO(model_path)
print(f"Model     : {model_path}")

# ── CSV setup ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(script_dir, args.csv)
CSV_HEADER = [
    "timestamp", "device_id", "source",
    "frame_id", "group_id",
    "n_potholes_in_group",
    "width_cm", "height_cm",
    "box_area_cm2", "pixel_area_cm2",
    "box_area_m2", "pixel_area_m2",      
    "repair_box_w_m", "repair_box_h_m", "repair_box_area_m2",
    "repair_label",
    "max_confidence", "method",
]
if not os.path.isfile(csv_path):
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


# ══════════════════════════════════════════════════════════════════════════════
#  Measurement helpers  (unchanged from still-image version)
# ══════════════════════════════════════════════════════════════════════════════

def get_contour_bbox(mask):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return {
        "left_col": float(x), "top_row": float(y),
        "right_col": float(x+w), "bot_row": float(y+h),
        "mid_row": float(y+h/2), "mid_col": float(x+w/2),
        "px_width": float(w), "px_height": float(h),
        "px_count": int(np.count_nonzero(mask)),
    }


def repair_box(width_m, height_m, step=0.5):
    box_w = math.ceil(width_m / step) * step
    box_h = math.ceil(height_m / step) * step
    return box_w, box_h, box_w * box_h


def get_pixel_repair_box(mask, rep_w_m, rep_h_m):
    bb = get_contour_bbox(mask)
    if bb is None:
        return None
    mid_row = int(np.clip(bb["mid_row"], 0, len(scale_w)-1))
    bot_row = int(np.clip(bb["bot_row"], 0, len(scale_h)-1))
    sw = scale_w[mid_row]
    sh = scale_h[bot_row]
    rep_w_px = (rep_w_m * 100) / sw if sw > 0 else 0
    rep_h_px = (rep_h_m * 100) / sh if sh > 0 else 0
    cx = (bb["left_col"] + bb["right_col"]) / 2
    cy = (bb["top_row"]  + bb["bot_row"])   / 2
    return {
        "px": (int(cx-rep_w_px/2), int(cy-rep_h_px/2),
               int(cx+rep_w_px/2), int(cy+rep_h_px/2)),
        "sw": sw, "sh": sh,
    }


def boxes_touch_or_overlap(a, b, gap=5):
    ax1,ay1,ax2,ay2 = a[0]-gap, a[1]-gap, a[2]+gap, a[3]+gap
    bx1,by1,bx2,by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def measure_ie(mask):
    bb = get_contour_bbox(mask)
    if bb is None:
        return np.nan, np.nan, np.nan, np.nan
    mid_row = int(np.clip(bb["mid_row"], 0, len(scale_w)-1))
    bot_row = int(np.clip(bb["bot_row"], 0, len(scale_h)-1))
    sw = scale_w[mid_row]
    sh = scale_h[bot_row]
    if np.isnan(sw) or np.isnan(sh):
        return np.nan, np.nan, np.nan, np.nan
    width_m      = (bb["px_width"]  * sw) / 100.0
    height_m     = (bb["px_height"] * sh) / 100.0
    box_area_m2  = width_m * height_m
    pixel_area_m2= (bb["px_count"] * sw * sh) / 10000.0
    return width_m, height_m, box_area_m2, pixel_area_m2


def merge_repair_boxes(detections):
    n = len(detections)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i+1, n):
            bi = detections[i]["repair_box_px"]
            bj = detections[j]["repair_box_px"]
            if bi and bj and boxes_touch_or_overlap(bi["px"], bj["px"]):
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    groups = []
    for indices in clusters.values():
        dets = [detections[i] for i in indices]
        if len(dets) == 1:
            d = dets[0]
            groups.append({
                "n_potholes"      : 1,
                "merged_px_box"   : d["repair_box_px"]["px"] if d["repair_box_px"] else (0,0,0,0),
                "merged_width_cm" : d["width_cm"],
                "merged_height_cm": d["height_cm"],
                "merged_width_m"  : d["width_m"],
                "merged_height_m" : d["height_m"],
                "repair_width_m"  : d["rep_w"],
                "repair_height_m" : d["rep_h"],
                "repair_area_m2"  : d["rep_area"],
                "repair_label"    : f"{d['rep_w']:.1f}m x {d['rep_h']:.1f}m",
                "total_area_cm2"  : d["box_area_m2"]*10000,
                "max_conf"        : d["conf"],
                "detections"      : dets,
            })
            continue

        all_px    = [d["repair_box_px"]["px"] for d in dets if d["repair_box_px"]]
        mx1 = min(p[0] for p in all_px); my1 = min(p[1] for p in all_px)
        mx2 = max(p[2] for p in all_px); my2 = max(p[3] for p in all_px)
        total_c   = sum(d["conf"] for d in dets) or 1.0
        sw_avg    = sum(d["repair_box_px"]["sw"]*d["conf"] for d in dets)/total_c
        sh_avg    = sum(d["repair_box_px"]["sh"]*d["conf"] for d in dets)/total_c
        mw_cm = (mx2-mx1)*sw_avg; mh_cm = (my2-my1)*sh_avg
        mw_m  = mw_cm/100;        mh_m  = mh_cm/100
        rw, rh, ra = repair_box(mw_m, mh_m)
        groups.append({
            "n_potholes"      : len(dets),
            "merged_px_box"   : (mx1,my1,mx2,my2),
            "merged_width_cm" : mw_cm, "merged_height_cm": mh_cm,
            "merged_width_m"  : mw_m,  "merged_height_m" : mh_m,
            "repair_width_m"  : rw,    "repair_height_m" : rh,
            "repair_area_m2"  : ra,
            "repair_label"    : f"{rw:.1f}m x {rh:.1f}m",
            "total_area_cm2"  : sum(d["box_area_m2"]*10000 for d in dets),
            "max_conf"        : max(d["conf"] for d in dets),
            "detections"      : dets,
        })
    groups.sort(key=lambda g: g["repair_area_m2"], reverse=True)
    return groups


# ══════════════════════════════════════════════════════════════════════════════
#  Per-frame processing  (used by both still and video paths)
# ══════════════════════════════════════════════════════════════════════════════

def process_frame(frame, source_name, frame_id, device_id):
    """
    Run detection + measurement + grouping on one BGR frame.
    Returns annotated frame and list of group dicts.
    """
    # Flip upside-down camera (skip with --no_flip)
    if not args.no_flip:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    img_undist = cv2.undistort(frame, K, dist)
    display    = img_undist.copy()
    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    results    = model(img_undist, task="segment",
                       conf=args.conf, verbose=False)
    detections = []

    for result in results:
        if result.masks is None:
            continue
        for mask_data, box in zip(result.masks.data, result.boxes):
            mask = mask_data.cpu().numpy()
            mask = cv2.resize(
                mask, (img_undist.shape[1], img_undist.shape[0]),
                interpolation=cv2.INTER_NEAREST).astype(bool)

            conf = float(box.conf)
            w_m, h_m, ba_m2, pa_m2 = measure_ie(mask)
            if np.isnan(w_m):
                continue

            rw, rh, ra  = repair_box(w_m, h_m)
            repair_px   = get_pixel_repair_box(mask, rw, rh)

            detections.append({
                "mask"         : mask,
                "box_xyxy"     : box.xyxy[0].cpu().numpy(),
                "conf"         : conf,
                "width_m"      : w_m,  "height_m"     : h_m,
                "width_cm"     : w_m*100, "height_cm"  : h_m*100,
                "box_area_m2"  : ba_m2,"pixel_area_m2": pa_m2,
                "rep_w"        : rw,   "rep_h"        : rh,
                "rep_area"     : ra,   "repair_box_px": repair_px,
            })

    groups = merge_repair_boxes(detections)

    # ── Draw individual detections ─────────────────────────────────────────
    for det in detections:
        overlay = np.zeros_like(display)
        overlay[det["mask"]] = (0, 255, 100)
        display = cv2.addWeighted(display, 1.0, overlay, 0.4, 0)
        x1,y1,x2,y2 = map(int, det["box_xyxy"])
        cv2.rectangle(display, (x1,y1), (x2,y2), (0,200,255), 1)
        if det["repair_box_px"]:
            rx1,ry1,rx2,ry2 = det["repair_box_px"]["px"]
            cv2.rectangle(display, (rx1,ry1), (rx2,ry2), (0,165,255), 1)
        lines = [
            f"W:{det['width_cm']:.1f}cm H:{det['height_cm']:.1f}cm",
            f"Area:{det['box_area_m2']*10000:.0f}cm2  conf:{det['conf']:.2f}",
        ]
        ty = max(y1 - len(lines)*20 - 5, 10)
        for j, line in enumerate(lines):
            yy = ty + j*20
            cv2.putText(display, line, (x1,yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
            cv2.putText(display, line, (x1,yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

    
    # ── Draw groups + log ──────────────────────────────────────────────────
    for g_idx, g in enumerate(groups):
        group_id = f"G{g_idx+1}"
        gx1,gy1,gx2,gy2 = g["merged_px_box"]
        cv2.rectangle(display, (gx1,gy1), (gx2,gy2), (0,0,220), 2)
        lines = [
            f"G{g_idx+1}: {g['n_potholes']} pothole(s)",
            f"Repair: {g['repair_label']}",
            f"Area: {g['repair_area_m2']:.2f} m2",
        ]
        ty = max(gy1 - len(lines)*22 - 5, 10)
        for j, line in enumerate(lines):
            yy = ty + j*22
            cv2.putText(display, line, (gx1,yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            cv2.putText(display, line, (gx1,yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,180), 1)

        # Frame counter overlay (top-left)
        cv2.putText(display,
                    f"Frame {frame_id}  |  {len(groups)} group(s)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 2)
        
        # Write each individual pothole in this group
        for pot_idx, det in enumerate(g["detections"]):
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    timestamp, device_id, source_name,
                    frame_id,
                    group_id,                          # ← maps pothole to group
                    1,                                 # n_potholes = 1 (individual)
                    f"{det['width_cm']:.1f}",
                    f"{det['height_cm']:.1f}",
                    f"{det['box_area_m2']*10000:.0f}",
                    f"{det['pixel_area_m2']*10000:.0f}",
                    f"{det['box_area_m2']:.4f}",
                    f"{det['pixel_area_m2']:.4f}",
                    det["rep_w"],
                    det["rep_h"],
                    f"{det['rep_area']:.4f}",
                    f"{det['rep_w']:.1f}m x {det['rep_h']:.1f}m",
                    f"{det['conf']:.2f}",
                    "IE",                              # ← individual rows use "IE"
                ])

        # Write the merged group row
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                timestamp, device_id, source_name,
                frame_id, group_id,
                g["n_potholes"],
                f"{g['merged_width_cm']:.1f}",
                f"{g['merged_height_cm']:.1f}",
                f"{g['total_area_cm2']:.0f}",
                "",                                     # pixel_area_cm2 (not applicable)
                f"{g['merged_width_m']:.4f}",
                "",                                     # pixel_area_m2 (not applicable)
                g["repair_width_m"], g["repair_height_m"],
                f"{g['repair_area_m2']:.4f}",
                g["repair_label"],
                f"{g['max_conf']:.2f}",
                f"IE_GROUP_{g['n_potholes']}",
            ])

    return display, groups


# ══════════════════════════════════════════════════════════════════════════════
#  Source handlers
# ══════════════════════════════════════════════════════════════════════════════

def run_image(img_path, device_id):
    """Process a single still image."""
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"ERROR: Cannot read image: {img_path}")

    base      = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(script_dir, f"{base}_annotated.jpg")

    display, groups = process_frame(
        img, source_name=base, frame_id=1, device_id=device_id)

    cv2.imwrite(save_path, display)
    print(f"\nSaved : {save_path}")
    print(f"CSV   : {csv_path}")

    if not args.no_display:
        cv2.imshow("Pothole Detection", display)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(source, device_id, is_camera=False):
    """
    Process a video file or live camera stream frame by frame.

    source     : file path (str) or camera index (int)
    is_camera  : True when reading from a live camera
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"ERROR: Cannot open video source: {source}")

    # Source label for CSV
    if is_camera:
        source_name = f"camera_{source}"
    else:
        source_name = os.path.splitext(os.path.basename(source))[0]

    # Video properties
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # -1 for live camera

    print(f"\nSource : {source_name}")
    print(f"Size   : {width}×{height}  FPS: {fps:.1f}")
    if total > 0:
        print(f"Frames : {total}")
    print(f"Skip   : every {args.frame_skip} frame(s)")
    print(f"Display: {'OFF (headless)' if args.no_display else 'ON'}")
    print("\nPress Q to quit.\n")

    # Output video writer
    writer = None
    if args.save_video or is_camera:
        out_name = os.path.join(
            script_dir, f"{source_name}_annotated_{today}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            out_name, fourcc,
            fps / max(args.frame_skip, 1),
            (width, height))
        print(f"Saving output video to: {out_name}")

    frame_idx     = 0
    processed_idx = 0
    total_groups  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for performance
        if (frame_idx - 1) % args.frame_skip != 0:
            if writer:
                writer.write(frame)   # write unprocessed frames as-is
            continue

        processed_idx += 1
        display, groups = process_frame(
            frame,
            source_name = source_name,
            frame_id    = frame_idx,
            device_id   = device_id,
        )
        total_groups += len(groups)

        if writer:
            writer.write(display)

        if not args.no_display:
            cv2.imshow("Pothole Detection  [Q to quit]", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user.")
                break

        # Progress print every 30 processed frames
        if processed_idx % 30 == 0:
            print(f"  Frame {frame_idx}  |  "
                  f"Groups this session: {total_groups}  |  "
                  f"CSV rows: {csv_path}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\nFinished.  Processed {processed_idx} frame(s).")
    print(f"Total repair groups logged : {total_groups}")
    print(f"CSV : {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device_id = args.device

    if args.image:
        print(f"\n[Mode] Still image → {args.image}")
        run_image(args.image, device_id)

    elif args.video:
        print(f"\n[Mode] Video file → {args.video}")
        run_video(args.video, device_id, is_camera=False)

    elif args.camera is not None:
        print(f"\n[Mode] Live camera → index {args.camera}")
        run_video(args.camera, device_id, is_camera=True)

    else:
        # Default fallback: still image in same folder as script
        default = os.path.join(script_dir, "Pothole_Grid_Img", "Pot_grid_12.jpg")
        print(f"\n[Mode] No input specified — using default image: {default}")
        run_image(default, device_id)
