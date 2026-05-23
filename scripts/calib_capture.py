import time
import os
from picamera2 import Picamera2

# Create folder
save_dir = "calibration_images_17"
os.makedirs(save_dir, exist_ok=True)

# Setup camera
picam2 = Picamera2()
config = picam2.create_still_configuration(
    main={"size": (1920, 1080), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)  # let camera warm up

num_images = 5
delay      = 5  # seconds to reposition

print("Starting in 3 seconds... reposition checkerboard between shots!")
time.sleep(3)

for i in range(num_images):
    print(f"Capturing image {i+1}/{num_images} in {delay} seconds...")
    time.sleep(delay)

    filename = os.path.join(save_dir, f"calib_{i+1:02d}.jpg")
    picam2.capture_file(filename)
    print(f"  Saved: {filename}")

picam2.stop()
print("Done! Check your calibration_images folder.")
