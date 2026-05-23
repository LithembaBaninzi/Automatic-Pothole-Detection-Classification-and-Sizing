# This is to validate the scale without the mask 

import cv2
import numpy as np
import os

# =====================================================
# FILES
# =====================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(script_dir, "calibration_images_17", "calib_01.jpg")
# Row scale tables
scale_w_lr = np.load(os.path.join(script_dir, "scale_table_w_lr_17_out.npy"))
scale_h_lr = np.load(os.path.join(script_dir, "scale_table_h_lr_17_out.npy"))

scale_w_ie = np.load(os.path.join(script_dir, "scale_table_w_ie_17_out.npy"))
scale_h_ie = np.load(os.path.join(script_dir, "scale_table_h_ie_17_out.npy"))


print("Loaded LR + IE scale tables")

# =====================================================
# LOAD IMAGE
# =====================================================
img = cv2.imread(IMAGE_PATH)
img = cv2.rotate(img, cv2.ROTATE_180)
base = img.copy()
display = img.copy()

# =====================================================
# GLOBALS
# =====================================================
points = []

# =====================================================
# DRAW FUNCTION
# =====================================================
def redraw():
    global display
    display = base.copy()

    for i, p in enumerate(points):
        cv2.circle(display, tuple(p), 5, (0,0,255), -1)

        if i > 0:
            cv2.line(display,
                     tuple(points[i-1]),
                     tuple(points[i]),
                     (255,0,0), 2)

    cv2.imshow("Pothole Select", display)

# =====================================================
# CLICK FUNCTION
# =====================================================
def click_event(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        redraw()

# =====================================================
# UI
# =====================================================
print("\nClick around pothole boundary")
print("ENTER = finish")
print("U = undo")
print("Q = quit")

cv2.namedWindow("Pothole Select", cv2.WINDOW_NORMAL)
cv2.imshow("Pothole Select", display)
cv2.setMouseCallback("Pothole Select", click_event)

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 13:      # ENTER
        break

    elif key == ord('u'):
        if len(points) > 0:
            points.pop()
            redraw()

    elif key == ord('q'):
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# =====================================================
# CHECK
# =====================================================
if len(points) < 3:
    print("Need at least 3 points.")
    exit()

pts = np.array(points, dtype=np.int32)

# =====================================================
# POLYGON AREA IN PIXELS
# =====================================================
def polygon_area_px(pts):
    x = pts[:,0]
    y = pts[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

# =====================================================
# ROW SCALE METHOD
# =====================================================
def measure_polygon(scale_w, scale_h):

    # Bounding box
    x_min = np.min(pts[:,0])
    x_max = np.max(pts[:,0])

    y_min = np.min(pts[:,1])
    y_max = np.max(pts[:,1])

    width_px  = x_max - x_min
    height_px = y_max - y_min

    mid_row = int((y_min + y_max) / 2)
    bot_row = int(y_max)

    mid_row = np.clip(mid_row, 0, len(scale_w)-1)
    bot_row = np.clip(bot_row, 0, len(scale_h)-1)

    # cm/px
    sw = scale_w[mid_row]
    sh = scale_h[bot_row]

    width_cm  = width_px  * sw
    height_cm = height_px * sh

    # Better area estimate using polygon pixels
    area_px = polygon_area_px(pts)
    area_cm2 = area_px * sw * sh

    return width_cm, height_cm, area_cm2

# =====================================================
# CALCULATE
# =====================================================
w_lr, h_lr, a_lr = measure_polygon(scale_w_lr, scale_h_lr)
w_ie, h_ie, a_ie = measure_polygon(scale_w_ie, scale_h_ie)
# w_lin, h_lin, a_lin = measure_polygon(scale_lin, scale_lin)
# w_exp, h_exp, a_exp = measure_polygon(scale_exp, scale_exp)

# =====================================================
# PRINT RESULTS
# =====================================================
print("\n==============================")
print("POTHOLE MEASUREMENTS")
print("==============================")

print("\nLinear Regression (LR)")
print(f"Width  : {w_lr:.2f} cm")
print(f"Height : {h_lr:.2f} cm")
print(f"Area   : {a_lr:.2f} cm²")

print("\nInterpolation/Extrapolation (IE)")
print(f"Width  : {w_ie:.2f} cm")
print(f"Height : {h_ie:.2f} cm")
print(f"Area   : {a_ie:.2f} cm²")



# =====================================================
# OPTIONAL VISUAL SAVE
# =====================================================
overlay = base.copy()
cv2.polylines(overlay, [pts], True, (0,255,0), 2)

cv2.putText(overlay,
            f"LR: {a_lr:.1f} cm2",
            (30,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2)

cv2.putText(overlay,
            f"IE: {a_ie:.1f} cm2",
            (30,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2)

save_path = os.path.join(script_dir, "pothole_compare_lr_ie_calib_img01_1m_17_out.jpg")
cv2.imwrite(save_path, overlay)

print(f"\nSaved: {save_path}")