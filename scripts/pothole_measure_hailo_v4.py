#!/usr/bin/env python3
"""
pothole_measure_hailo_v4.py
============================
Final pipeline for RPi5 + Hailo-8 + RPi camera.

What's new vs v3:
  1. SORT-style Kalman tracker (hailotracker equivalent in pure Python)
     • Predicts each pothole's next position from its velocity
     • Maintains tracks through brief misdetections
     • Far fewer duplicate IDs than the simple IoU tracker

  2. Camera lens undistortion using K_matlab1.npy + dist_matlab1.npy
     • Pre-computed remap (~1ms per frame, vs ~15ms for cv2.undistort)
     • Same calibration as your PC scripts

  3. RPi camera support via picamera2
     • --input rpi          → standard libcamera capture
     • Works with both the IMX500 AI camera and standard CSI cameras

  4. All v3 optimisations preserved
     • Batch inference, reader thread, async crop saving
     • No drawing unless --save_video
     • Correct scale-table row mapping

Usage:
    # Live RPi camera, log only
    python pothole_measure_hailo_v4.py --input rpi --no_flip

    # Video file, save annotated output
    python pothole_measure_hailo_v4.py --input p.mp4 --no_flip --save_video

    # USB webcam with GPS
    python pothole_measure_hailo_v4.py --input 0 --gps --weather "Sunny"

    # Disable undistortion if needed
    python pothole_measure_hailo_v4.py --input p.mp4 --no_undistort
"""

import csv, os, sys, math, argparse, threading, time, re, queue
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import numpy as np
import cv2

try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface,
        InferVStreams, ConfigureParams,
        InputVStreamParams, OutputVStreamParams, FormatType)
except ImportError:
    sys.exit("ERROR: activate the Hailo venv first.")

try:
    import serial
except ImportError:
    serial = None

# ── Paths + calibration ──────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
files_dir  = os.path.join(script_dir, "files")

scale_w = np.load(os.path.join(files_dir, "scale_table_w_ie_17_out.npy"))
scale_h = np.load(os.path.join(files_dir, "scale_table_h_ie_17_out.npy"))
TABLE_H = len(scale_w)
print(f"[Calib] Scale_w: {np.nanmin(scale_w):.4f}–{np.nanmax(scale_w):.4f} cm/px "
      f"({TABLE_H} rows)")
print(f"[Calib] Scale_h: {np.nanmin(scale_h):.4f}–{np.nanmax(scale_h):.4f} cm/px")

CSV_HEADER = [
    "timestamp", "device_id", "source", "frame_id", "track_id",
    "n_potholes_in_group",
    "width_cm", "height_cm", "box_area_cm2", "pixel_area_cm2",
    "box_area_m2", "pixel_area_m2",
    "repair_box_w_m", "repair_box_h_m", "repair_box_area_m2", "repair_label",
    "confidence", "method",
    "centroid_u", "centroid_v",
    "repair_box_x1", "repair_box_y1", "repair_box_x2", "repair_box_y2",
    "gps_lat", "gps_lon", "alt_m", "speed_km", "gps_utc",
    "weather", "image_file",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Camera lens undistortion (precomputed remap → fast per-frame)
# ══════════════════════════════════════════════════════════════════════════════

class CameraUndistorter:
    """
    Pre-computes the inverse mapping once, then cv2.remap is ~1ms per frame.
    Falls back to identity (no-op) if calibration files are missing.
    """
    def __init__(self, K_path, dist_path, size, enabled=True):
        self.enabled = False
        if not enabled:
            print("[Undistort] disabled by --no_undistort")
            return
        if not (os.path.isfile(K_path) and os.path.isfile(dist_path)):
            print(f"[Undistort] disabled — calibration files not found:")
            print(f"            {K_path}")
            print(f"            {dist_path}")
            return
        K    = np.load(K_path)
        dist = np.load(dist_path)
        W, H = size
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0)
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, (W, H), cv2.CV_32FC1)
        self.enabled = True
        print(f"[Undistort] enabled  fx={K[0,0]:.1f}  size={W}x{H}")

    def apply(self, frame):
        if not self.enabled:
            return frame
        return cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)


# ══════════════════════════════════════════════════════════════════════════════
#  YOLOv8-seg post-processing (proven from v3)
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def decode_dfl(raw, num_bins=16):
    n = raw.shape[0]
    raw  = raw.reshape(n, 4, num_bins)
    dist = softmax(raw, axis=-1)
    return np.sum(dist * np.arange(num_bins, dtype=np.float32), axis=-1)

def nms(boxes, scores, iou_threshold=0.7):
    if len(boxes) == 0: return []
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas  = (x2-x1)*(y2-y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou   = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order = order[np.where(iou<=iou_threshold)[0]+1]
    return keep

def classify_outputs(output_dict):
    arrays = {}
    for name, data in output_dict.items():
        arr = np.array(data)
        if arr.ndim == 4: arr = arr[0]
        if arr.ndim == 2: arr = arr[:,:,np.newaxis]
        if arr.ndim == 3:
            d0,d1,d2 = arr.shape
            if d0 < d1 and d0 < d2 and d0 != d1:
                arr = arr.transpose(1,2,0)
        arrays[name] = arr
    by_size = {}
    for name,arr in arrays.items():
        by_size.setdefault(arr.shape[0],[]).append((name,arr))
    sizes = sorted(by_size.keys(), reverse=True)
    proto = [a for _,a in by_size.pop(sizes[0]) if a.shape[-1]==32][0]
    heads = []
    for sz in sorted(by_size.keys(), reverse=True):
        stride = 640 // sz
        ld = {}
        for _,arr in by_size[sz]:
            c = arr.shape[-1]
            if   c==64: ld["bbox"] = arr
            elif c==1:  ld["cls"]  = arr
            elif c==32: ld["mask_coeff"] = arr
        if all(k in ld for k in ("bbox","cls","mask_coeff")):
            heads.append((stride, ld["bbox"], ld["cls"], ld["mask_coeff"]))
    return heads, proto

def run_postprocess(output_dict, orig_shape,
                    conf_thresh=0.35, iou_thresh=0.7, mask_thresh=0.5):
    orig_h, orig_w = orig_shape[:2]
    heads, proto   = classify_outputs(output_dict)
    if not heads: return []

    all_boxes, all_scores, all_mcoeffs = [], [], []
    for stride, bbox_raw, cls_raw, mcoeff_raw in heads:
        h, w = bbox_raw.shape[:2]
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        cx = (gx.flatten() + 0.5) * stride
        cy = (gy.flatten() + 0.5) * stride
        bbox_flat   = bbox_raw.reshape(-1, 64)
        cls_flat    = cls_raw.reshape(-1)
        mcoeff_flat = mcoeff_raw.reshape(-1, 32)
        dist = decode_dfl(bbox_flat)
        x1 = cx - dist[:,0]*stride; y1 = cy - dist[:,1]*stride
        x2 = cx + dist[:,2]*stride; y2 = cy + dist[:,3]*stride
        boxes  = np.stack([x1,y1,x2,y2], axis=-1)
        scores = cls_flat if (cls_flat.min()>=0 and cls_flat.max()<=1) \
                 else sigmoid(cls_flat)
        keep = scores >= conf_thresh
        if not np.any(keep): continue
        all_boxes.append(boxes[keep]); all_scores.append(scores[keep])
        all_mcoeffs.append(mcoeff_flat[keep])
    if not all_boxes: return []

    boxes   = np.concatenate(all_boxes)
    scores  = np.concatenate(all_scores)
    mcoeffs = np.concatenate(all_mcoeffs)
    keep_idx = nms(boxes, scores, iou_thresh)
    boxes, scores, mcoeffs = boxes[keep_idx], scores[keep_idx], mcoeffs[keep_idx]

    ph, pw = proto.shape[:2]
    proto_flat  = proto.reshape(-1, 32)
    mask_logits = (mcoeffs @ proto_flat.T).reshape(-1, ph, pw)
    mask_probs  = sigmoid(mask_logits)

    sx, sy = orig_w / 640, orig_h / 640
    results = []
    for i in range(len(boxes)):
        bx1 = max(0, int(boxes[i,0]*sx)); by1 = max(0, int(boxes[i,1]*sy))
        bx2 = min(orig_w, int(boxes[i,2]*sx)); by2 = min(orig_h, int(boxes[i,3]*sy))
        m = cv2.resize(mask_probs[i], (orig_w, orig_h),
                       interpolation=cv2.INTER_LINEAR)
        full_mask = np.zeros((orig_h, orig_w), dtype=bool)
        full_mask[by1:by2, bx1:bx2] = m[by1:by2, bx1:bx2] > mask_thresh
        if np.count_nonzero(full_mask) < 10: continue
        results.append({"box":(bx1,by1,bx2,by2),
                        "score":float(scores[i]), "mask":full_mask})
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  IE measurement
# ══════════════════════════════════════════════════════════════════════════════

def repair_box(width_m, height_m, step=0.5):
    w = math.ceil(max(width_m, 0.001) / step) * step
    h = math.ceil(max(height_m, 0.001) / step) * step
    return w, h, w * h

def measure_pothole(mask, frame_h):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 1 or h < 1: return None
    mid_row = int(np.clip((y + h/2) * TABLE_H / frame_h, 0, TABLE_H - 1))
    bot_row = int(np.clip((y + h)   * TABLE_H / frame_h, 0, TABLE_H - 1))
    sw, sh  = scale_w[mid_row], scale_h[bot_row]
    if np.isnan(sw) or np.isnan(sh): return None
    width_cm  = w * sw
    height_cm = h * sh
    px_count  = int(np.count_nonzero(mask))
    M  = cv2.moments(cnt)
    cx = M["m10"]/M["m00"] if M["m00"] else x + w/2
    cy = M["m01"]/M["m00"] if M["m00"] else y + h/2
    return {"width_cm":width_cm, "height_cm":height_cm,
            "box_area_cm2":width_cm*height_cm,
            "pixel_area_cm2":px_count*sw*sh,
            "box_area_m2":width_cm*height_cm/10000,
            "pixel_area_m2":px_count*sw*sh/10000,
            "cx":cx, "cy":cy, "sw":sw, "sh":sh}

def get_pixel_repair_box(cx, cy, rep_w_m, rep_h_m, sw, sh, H, W):
    w_px = (rep_w_m*100)/sw if sw > 0 else 0
    h_px = (rep_h_m*100)/sh if sh > 0 else 0
    x1 = max(0, min(int(cx - w_px/2), W-1))
    y1 = max(0, min(int(cy - h_px/2), H-1))
    x2 = max(0, min(int(cx + w_px/2), W-1))
    y2 = max(0, min(int(cy + h_px/2), H-1))
    return x1, y1, x2, y2


# ══════════════════════════════════════════════════════════════════════════════
#  SORT-style Kalman tracker  (hailotracker equivalent in pure Python)
# ══════════════════════════════════════════════════════════════════════════════

class KalmanBoxTracker:
    """
    Single-object Kalman filter for SORT-style tracking.
    State: [cx, cy, scale, ratio, dcx, dcy, dscale]  (7-dim)
      - cx, cy : centre coordinates
      - scale  : box area (w x h)
      - ratio  : aspect (w / h)
      - d*     : velocities (constant-velocity motion model)
    """
    _next_id = 1

    def __init__(self, bbox):
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        s  = w * h
        r  = w / h if h > 0 else 1.0

        self.x = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(7, dtype=np.float32) * 10.0

        # State transition (constant velocity for cx, cy, scale)
        self.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)

        # Measurement matrix (we observe cx, cy, scale, ratio)
        self.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.float32)

        self.R = np.eye(4, dtype=np.float32) * 1.0    # measurement noise
        self.Q = np.eye(7, dtype=np.float32) * 0.01   # process noise
        self.Q[4:, 4:] *= 0.1                          # velocities are smoother

        self.time_since_update = 0
        self.hits = 1
        self.age  = 1

    def predict(self):
        """Advance state forward by one frame."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if self.x[2] < 1: self.x[2] = 1   # scale must stay positive
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox):
        """Correct state with new detection."""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        z  = np.array([cx, cy, w*h, w/h if h>0 else 1.0], dtype=np.float32)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7, dtype=np.float32) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1

    def get_box(self):
        """Return current state as xyxy bounding box."""
        cx, cy, s, r = float(self.x[0]), float(self.x[1]), \
                       float(self.x[2]), float(self.x[3])
        s = max(s, 1.0); r = max(r, 0.01)
        w = math.sqrt(s * r)
        h = s / w
        return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


class SORTTracker:
    """
    SORT-style multi-object tracker.
    Equivalent to Hailo's hailotracker but in pure Python.
    """
    def __init__(self, iou_thresh=0.3, max_lost=15, min_hits=1):
        self.iou_thresh = iou_thresh
        self.max_lost   = max_lost
        self.min_hits   = min_hits
        self.trackers   = []   # list[KalmanBoxTracker]

    @staticmethod
    def _iou(a, b):
        ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
        ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
        ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
        inter = max(0,ix2-ix1) * max(0,iy2-iy1)
        aa = max(0, (ax2-ax1)) * max(0, (ay2-ay1))
        ab = max(0, (bx2-bx1)) * max(0, (by2-by1))
        return inter / (aa + ab - inter + 1e-6)

    def update(self, detections):
        """
        detections: list of dicts with 'box' (xyxy).
        Returns: list of track IDs aligned with detections.
        """
        # 1. Predict all existing tracks forward
        for t in self.trackers:
            t.predict()

        N = len(detections)
        M = len(self.trackers)
        assigned = [None] * N

        # 2. Match detections to predicted tracks (greedy by IoU)
        if N and M:
            predicted = [t.get_box() for t in self.trackers]
            pairs = []
            for i, det in enumerate(detections):
                for j, pred in enumerate(predicted):
                    iou = self._iou(det['box'], pred)
                    if iou >= self.iou_thresh:
                        pairs.append((iou, i, j))
            pairs.sort(reverse=True)

            matched_dets = set()
            matched_trks = set()
            for iou, i, j in pairs:
                if i in matched_dets or j in matched_trks: continue
                matched_dets.add(i); matched_trks.add(j)
                self.trackers[j].update(detections[i]['box'])
                assigned[i] = self.trackers[j].id
        else:
            matched_dets = set()

        # 3. New tracks for unmatched detections
        for i in range(N):
            if i not in matched_dets:
                t = KalmanBoxTracker(detections[i]['box'])
                self.trackers.append(t)
                assigned[i] = t.id

        # 4. Drop old tracks
        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.max_lost]
        return assigned


# ══════════════════════════════════════════════════════════════════════════════
#  GPS reader  (unchanged from v3)
# ══════════════════════════════════════════════════════════════════════════════

class GPSReader:
    def __init__(self, port="/dev/ttyUSB2", baud=115200):
        self.ser=None; self._last_fix=None; self._running=True
        if serial is None: return
        try:
            self.ser=serial.Serial(port,baud,timeout=2)
            self.ser.write(b"AT+CGPS=1\r\n"); time.sleep(0.5); self.ser.read(100)
            threading.Thread(target=self._loop, daemon=True).start()
            print(f"[GPS] enabled on {port}")
        except Exception as e:
            print(f"[GPS] not available: {e}")

    def _loop(self):
        while self._running:
            try:
                self.ser.write(b"AT+CGPSINFO\r\n"); time.sleep(1)
                resp = self.ser.read(500).decode(errors="replace")
                for line in resp.splitlines():
                    if line.startswith("+CGPSINFO:"):
                        fix = self._parse(line)
                        if fix: self._last_fix = fix; break
            except: pass
            time.sleep(2)

    def _parse(self, line):
        m = re.search(r'\+CGPSINFO:\s*(\d+\.\d+),([NS]),(\d+\.\d+),([EW]),'
                      r'(\d*),([\d\.]*),([\d\.]*),([\d\.]*)', line)
        if not m: return None
        lr,ld,lor,lod,date,utc,alt,spd = m.groups()
        lat = float(lr[:2])+float(lr[2:])/60; lon=float(lor[:3])+float(lor[3:])/60
        if ld=="S": lat=-lat
        if lod=="W": lon=-lon
        return {"lat":round(lat,6),"lon":round(lon,6),
                "alt_m":float(alt) if alt else None,
                "speed_km":float(spd)*1.852 if spd else None,
                "date":date or None,"time_utc":utc or None}

    def get_full_fix(self): return self._last_fix
    def close(self):
        self._running=False
        if self.ser and self.ser.is_open: self.ser.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Unified video source: file / USB cam / RPi cam (incl. AI camera IMX500)
# ══════════════════════════════════════════════════════════════════════════════

class VideoSource:
    def __init__(self, source_spec, width=1280, height=720):
        self.source = source_spec
        self.cap    = None
        self.picam  = None

        if isinstance(source_spec, str) and source_spec.lower() == "rpi":
            self._init_picamera(width, height)
            self.label = "rpi_camera"
        else:
            try:
                cam_idx = int(source_spec)
                self.cap = cv2.VideoCapture(cam_idx)
                self.label = f"camera_{cam_idx}"
            except (ValueError, TypeError):
                self.cap = cv2.VideoCapture(source_spec)
                self.label = os.path.splitext(os.path.basename(source_spec))[0]
            if not self.cap or not self.cap.isOpened():
                sys.exit(f"ERROR: cannot open source: {source_spec}")

    def _init_picamera(self, w, h):
        try:
            from picamera2 import Picamera2
        except ImportError:
            sys.exit("ERROR: picamera2 not installed.\n"
                     "  Install with: sudo apt install -y python3-picamera2")
        self.picam = Picamera2()
        config = self.picam.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"})
        self.picam.configure(config)
        self.picam.start()
        time.sleep(0.5)   # auto-exposure settle
        print(f"[Camera] picamera2 started @ {w}x{h}")

    def read(self):
        if self.picam:
            frame = self.picam.capture_array()  # RGB888
            # return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        return self.cap.read()

    def get_size(self):
        if self.picam:
            cfg = self.picam.camera_configuration()
            return cfg["main"]["size"]
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def get_fps(self):
        if self.picam: return 30.0
        return self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def get_total_frames(self):
        if self.picam or self.cap is None: return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def is_live(self):
        return self.picam is not None or self.get_total_frames() == 0

    def release(self):
        if self.cap: self.cap.release()
        if self.picam: self.picam.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  Drawing (only when --save_video)
# ══════════════════════════════════════════════════════════════════════════════
def draw_detections(frame, detections, track_ids, meas_list, rep_boxes):
    H, W = frame.shape[:2]

    # Combined mask overlay
    combined = np.zeros((H, W), dtype=bool)
    for det in detections:
        combined |= det["mask"]
    overlay = np.zeros_like(frame)
    overlay[combined] = (0, 255, 100)          # green
    cv2.addWeighted(frame, 1.0, overlay, 0.4, 0, frame)

    for det, tid, meas, rep in zip(detections, track_ids, meas_list, rep_boxes):
        if meas is None:
            continue

        # Detection bounding box (yellow)
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # Individual repair box (orange)
        rx1, ry1, rx2, ry2 = rep
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)

        # Repair dimensions
        rep_w, rep_h, _ = repair_box(meas['width_cm'] / 100, meas['height_cm'] / 100)

        # --- Outside text (above the detection box) ---
        outside_lines = [
            f"ID:{tid}",
            f"Conf:{det['score']:.2f}",
            f"Repair:{rep_w:.1f}m x {rep_h:.1f}m",
        ]
        tx_out = x1
        ty_out = max(y1 - len(outside_lines) * 18 - 5, 10)   # above box
        for j, line in enumerate(outside_lines):
            yy = ty_out + j * 18
            cv2.putText(frame, line, (tx_out, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(frame, line, (tx_out, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # --- Inside text (over the detection box) ---
        inside_lines = [
            f"{meas['width_cm']:.1f} x {meas['height_cm']:.1f}cm",
            f"BoxArea:{meas['box_area_cm2']:.0f}cm2",
        ]
        tx_in = x1 + 5
        ty_in = y1 + 18   # first line just below the top edge of the box
        for j, line in enumerate(inside_lines):
            yy = ty_in + j * 18
            cv2.putText(frame, line, (tx_in, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(frame, line, (tx_in, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return frame

# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Pothole measurement v4 — final")
    p.add_argument("--input", required=True,
                   help="Video file path, USB camera index (0/1/...), or 'rpi'")
    p.add_argument("--hef", default=os.path.join(files_dir,"pothole_seg_detector.hef"))
    p.add_argument("--device_id", default="RPi_001")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--frame_skip", type=int, default=1)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--no_flip",       action="store_true",
                   help="Skip 180° rotation (camera mounted normally)")
    p.add_argument("--no_undistort",  action="store_true",
                   help="Skip lens undistortion")
    p.add_argument("--save_video",    action="store_true")
    p.add_argument("--gps",           action="store_true")
    p.add_argument("--gps_port",      default="/dev/ttyUSB2")
    p.add_argument("--gps_baud",      type=int, default=115200)
    p.add_argument("--weather",       default="")
    p.add_argument("--cam_width",     type=int, default=1920,
                   help="Camera capture width (RPi/USB camera only)")
    p.add_argument("--cam_height",    type=int, default=1080,
                   help="Camera capture height (RPi/USB camera only)")
    p.add_argument("--K",    default=os.path.join(files_dir, "K_matlab1.npy"),
                   help="Camera intrinsics .npy")
    p.add_argument("--dist", default=os.path.join(files_dir, "dist_matlab1.npy"),
                   help="Distortion coefficients .npy")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Open source ──────────────────────────────────────────────────────
    src = VideoSource(args.input, args.cam_width, args.cam_height)
    vid_w, vid_h = src.get_size()
    fps_in       = src.get_fps()
    total_fr     = src.get_total_frames()
    is_live      = src.is_live()
    source_name  = src.label

    # ── Undistorter ──────────────────────────────────────────────────────
    undist = CameraUndistorter(args.K, args.dist, (vid_w, vid_h),
                               enabled=not args.no_undistort)

    # ── CSV + crops ──────────────────────────────────────────────────────
    csv_path  = os.path.join(script_dir, f"pothole_detections_{ts}.csv")
    crops_dir = os.path.join(script_dir, f"pothole_crops_{ts}")
    os.makedirs(crops_dir, exist_ok=True)
    csv_file   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_HEADER)

    # ── GPS ──────────────────────────────────────────────────────────────
    gps = GPSReader(args.gps_port, args.gps_baud) if args.gps else None

    # ── HEF ──────────────────────────────────────────────────────────────
    if not os.path.isfile(args.hef):
        sys.exit(f"ERROR: HEF not found: {args.hef}")
    hef        = HEF(args.hef)
    input_info = hef.get_input_vstream_infos()[0]
    in_h, in_w = input_info.shape[0], input_info.shape[1]

    print(f"[Model]   {args.hef}  input={in_w}×{in_h}")
    print(f"[Source]  {source_name}  {vid_w}×{vid_h}  {fps_in:.1f}fps  "
          f"{'(live)' if is_live else f'frames={total_fr}'}")
    print(f"[Tracker] SORT/Kalman (hailotracker equivalent)")
    print(f"[Optim]   batch={args.batch}  frame_skip={args.frame_skip}  "
          f"draw={'ON' if args.save_video else 'OFF'}  "
          f"undistort={'ON' if undist.enabled else 'OFF'}")
    print(f"[Output]  CSV → {csv_path}")

    video_writer  = None
    video_out_path = None
    if args.save_video:
        video_out_path = os.path.join(script_dir, f"pothole_annotated_{ts}.mp4")
        video_writer = cv2.VideoWriter(
            video_out_path, cv2.VideoWriter_fourcc(*"mp4v"),
            fps_in / max(args.frame_skip, 1), (vid_w, vid_h))
        print(f"[Output]  Video → {video_out_path}")

    tracker         = SORTTracker(iou_thresh=0.3, max_lost=15)
    saved_track_ids = set()

    # ── Reader thread ────────────────────────────────────────────────────
    frame_queue = queue.Queue(maxsize=args.batch * 3)
    stop_event  = threading.Event()

    def reader():
        idx = 0
        while not stop_event.is_set():
            ret, frame = src.read()
            if not ret:
                frame_queue.put(None); break
            idx += 1
            if (idx - 1) % args.frame_skip != 0:
                continue
            if not args.no_flip:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            # ── Undistort BEFORE inference (so model sees corrected geometry)
            frame = undist.apply(frame)
            img_rgb = cv2.cvtColor(
                cv2.resize(frame, (in_w, in_h)), cv2.COLOR_BGR2RGB)
            frame_queue.put((idx, frame, img_rgb))
        # If live source, the loop above won't terminate naturally
        if is_live:
            frame_queue.put(None)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    crop_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="crop_")
    def _write_crop(path, img): cv2.imwrite(path, img)

    print("\nProcessing... (Ctrl-C to stop)\n")
    frame_idx      = 0
    proc_count     = 0
    det_total      = 0
    t_start        = time.time()
    _last_progress = 0

    try:
        with VDevice() as target:
            cfg = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe)
            ng  = target.configure(hef, cfg)[0]
            in_params  = InputVStreamParams.make(
                ng, quantized=True, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(
                ng, quantized=False, format_type=FormatType.FLOAT32)

            with ng.activate(ng.create_params()):
                with InferVStreams(ng, in_params, out_params) as pipeline:

                    pending_frames  = []
                    pending_preproc = []

                    def flush_batch():
                        nonlocal proc_count, det_total, _last_progress
                        if not pending_frames: return
                        B = len(pending_frames)
                        batch = np.stack(pending_preproc)
                        raw = pipeline.infer({input_info.name: batch})

                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        gps_fix = gps.get_full_fix() if gps else None

                        for i, (fidx, orig_frame) in enumerate(pending_frames):
                            proc_count += 1
                            frame_out = {k: v[i:i+1] for k,v in raw.items()}
                            detections = run_postprocess(
                                frame_out, orig_frame.shape,
                                conf_thresh=args.conf)

                            track_ids = tracker.update(detections)
                            det_total += len(detections)

                            meas_list = []
                            rep_boxes = []
                            for det, tid in zip(detections, track_ids):
                                meas = measure_pothole(
                                    det["mask"], orig_frame.shape[0])
                                meas_list.append(meas)
                                if meas is None:
                                    rep_boxes.append((0,0,0,0)); continue

                                rep_w, rep_h, rep_area = repair_box(
                                    meas["width_cm"]/100,
                                    meas["height_cm"]/100)
                                rep_px = get_pixel_repair_box(
                                    meas["cx"], meas["cy"],
                                    rep_w, rep_h,
                                    meas["sw"], meas["sh"],
                                    orig_frame.shape[0], orig_frame.shape[1])
                                rep_boxes.append(rep_px)

                                # Async crop on first sighting
                                img_file = ""
                                if tid not in saved_track_ids:
                                    saved_track_ids.add(tid)
                                    rx1,ry1,rx2,ry2 = rep_px
                                    margin = 30
                                    H,W = orig_frame.shape[:2]
                                    cy1=max(0,ry1-margin); cy2=min(H,ry2+margin)
                                    cx1=max(0,rx1-margin); cx2=min(W,rx2+margin)
                                    crop = orig_frame[cy1:cy2, cx1:cx2].copy()
                                    if crop.size > 0:
                                        ov = np.zeros_like(crop)
                                        mc = det["mask"][cy1:cy2, cx1:cx2]
                                        ov[mc] = (0,255,100)
                                        crop = cv2.addWeighted(crop,1.0,ov,0.4,0)
                                        fname = os.path.join(
                                            crops_dir,
                                            f"pothole_track{tid}.jpg")
                                        crop_executor.submit(_write_crop, fname, crop)
                                        img_file = fname

                                rx1,ry1,rx2,ry2 = rep_px
                                csv_writer.writerow([
                                    now_str, args.device_id, source_name,
                                    fidx, tid, 1,
                                    f"{meas['width_cm']:.1f}",
                                    f"{meas['height_cm']:.1f}",
                                    f"{meas['box_area_cm2']:.0f}",
                                    f"{meas['pixel_area_cm2']:.0f}",
                                    f"{meas['box_area_m2']:.4f}",
                                    f"{meas['pixel_area_m2']:.4f}",
                                    f"{rep_w:.1f}", f"{rep_h:.1f}",
                                    f"{rep_area:.4f}",
                                    f"{rep_w:.1f}m x {rep_h:.1f}m",
                                    f"{det['score']:.2f}", "IE_KAL",
                                    f"{meas['cx']:.1f}", f"{meas['cy']:.1f}",
                                    rx1,ry1,rx2,ry2,
                                    gps_fix['lat'] if gps_fix else None,
                                    gps_fix['lon'] if gps_fix else None,
                                    gps_fix['alt_m'] if gps_fix else None,
                                    gps_fix['speed_km'] if gps_fix else None,
                                    (f"{gps_fix['date']}{gps_fix['time_utc']}"
                                     if gps_fix else None),
                                    args.weather, img_file,
                                ])
                            csv_file.flush()

                            if video_writer and args.save_video:
                                draw_detections(orig_frame, detections,
                                               track_ids, meas_list, rep_boxes)
                                cv2.putText(
                                    orig_frame,
                                    f"Frame {fidx}  {len(detections)} det  "
                                    f"{len(saved_track_ids)} tracks",
                                    (10,28), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55, (200,200,200), 2)
                                video_writer.write(orig_frame)
                                #frame_filename = os.path.join(frames_dir, f"frame_{fidx:06d}.jpg")
                               # cv2.imwrite(frame_filename, orig_frame)

                        last_fidx = pending_frames[-1][0]
                        pending_frames.clear()
                        pending_preproc.clear()

                        if proc_count - _last_progress >= 20:
                            _last_progress = proc_count
                            elapsed = time.time() - t_start
                            fps = proc_count / elapsed if elapsed > 0 else 0
                            pct = (last_fidx/total_fr*100) if total_fr > 0 else 0
                            print(f"  Frame {last_fidx:>5d}"
                                  f"{('/' + str(total_fr)):>5s}"
                                  f" ({pct:5.1f}%)  |  {fps:5.1f} fps"
                                  f"  |  tracks: {len(saved_track_ids)}"
                                  f"  |  dets: {det_total}")

                    while True:
                        try:
                            item = frame_queue.get(timeout=10)
                        except queue.Empty:
                            break
                        if item is None:
                            flush_batch(); break

                        fidx, orig, preproc = item
                        frame_idx = fidx
                        pending_frames.append((fidx, orig))
                        pending_preproc.append(preproc)
                        if len(pending_frames) >= args.batch:
                            flush_batch()

    except KeyboardInterrupt:
        print("\n[Stop] Interrupted.")
    except Exception as e:
        if "HAILO_OUT_OF_PHYSICAL_DEVICES" in str(e):
            sys.exit("ERROR: Hailo device locked — sudo systemctl restart hailort.service")
        raise
    finally:
        stop_event.set()
        crop_executor.shutdown(wait=True)
        reader_thread.join(timeout=3)
        src.release()
        if video_writer: video_writer.release()
        csv_file.close()
        if gps: gps.close()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Finished in {elapsed:.1f}s  ({proc_count} frames processed)")
    print(f"  Effective FPS             : {proc_count/elapsed:.1f}")
    print(f"  Unique potholes (Kalman)  : {len(saved_track_ids)}")
    print(f"  Total detections logged   : {det_total}")
    print(f"  CSV    : {csv_path}")
    print(f"  Crops  : {crops_dir}")
    if video_out_path: print(f"  Video  : {video_out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
