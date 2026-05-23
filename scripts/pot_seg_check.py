from ultralytics import YOLO
import os
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
K = os.path.join(script_dir, "best_seg.pt")
model = YOLO(K)
# results = model('test_pothole.jpg')

test_img  = os.path.join(script_dir,"Pothole_Grid_Img", "Pot_grid_14.jpg")
img        = cv2.imread(test_img)
img        = cv2.rotate(img, cv2.ROTATE_180)   # flip upside-down camera
save_img  = os.path.join(script_dir, "pothole_best_seg_grid14.jpg")
results = model(img)
results[0].show()  # visualise masks
results[0].save(save_img)