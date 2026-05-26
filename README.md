# 🚧 Automatic Pothole Detection, Classification & Sizing


Real‑time pothole detection and metric sizing using a **Raspberry Pi 5 + Hailo‑8 NPU**, a custom‑trained **YOLOv8‑seg** model, and an empirical **Interpolation/Extrapolation (IE) scale table** for lane‑marking‑free calibration.

**Live Demo Dashboard** → [https://lithemba.pythonanywhere.com/](https://lithemba.pythonanywhere.com/)  
> ⚠️ *This dashboard is hosted on PythonAnywhere’s free plan and may be deactivated after **31 August 2026**. If the link is no longer accessible, please refer to the sample output files in this repository.*

---

## 📌 Table of Contents
- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software & Dependencies](#software--dependencies)
- [Installation](#installation)
- [Calibration](#calibration)
- [Model Training](#model-training)
- [Running the Measurement Pipeline](#running-the-measurement-pipeline)
- [Video Input & Output Examples](#video-input--output-examples)
- [Results & Report](#results--report)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## 📖 Overview

This project implements a complete pipeline for automatic pothole detection and metric sizing on a low‑cost edge device (Raspberry Pi 5 with Hailo‑8 accelerator). Key features:

- **YOLOv8‑seg** instance segmentation model (trained on custom dataset, 91.6% mAP50)
- **Interpolation/Extrapolation (IE) scale table** – empirical pixel‑to‑metre conversion without lane markings
- **SORT‑based Kalman tracking** for consistent object IDs across frames
- **Optional GPS logging** (SIM7600X) and weather tagging
- **Annotated video output** with bounding boxes, repair boxes, and dimensions
- **Web dashboard** for viewing and exporting results

The system achieves **~13 FPS** on the Pi 5 + Hailo‑8, with average width error **<7%** and height error **<8%** on static validation objects.

---

## 🖥️ Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Raspberry Pi 5 | 8 GB RAM, 2.4 GHz quad‑core |
| Hailo‑8 AI Accelerator | M.2 module, 26 TOPS |
| RPi AI Camera (IMX500) | 12 MP, 66° FoV |
| SIM7600X 4G/GPS Module | (optional) for geotagging |
| Power supply | 5 V / 5 A USB‑C |

---

## 📦 Software & Dependencies

# Key Libraries

- **ultralytics** – YOLOv8‑seg model
- **opencv-python** – image processing, undistortion
- **numpy** – numerical operations
- **hailo_platform** – Hailo‑8 inference (requires HailoRT ≥4.23.0)
- **picamera2** – RPi camera capture
- **serial** – GPS communication

## 🛠️ Installation

### 1. Install HailoRT (on Raspberry Pi)
```bash
sudo apt install hailo-all -y
sudo reboot
```

### 2. Clone the repository
```bash
git clone https://github.com/LithembaBaninzi/Automatic-Pothole-Detection-Classification-and-Sizing.git
cd Automatic-Pothole-Detection-Classification-and-Sizing
```

### 3. Virtual Environment Setup for Hailo‑8 NPU on Raspberry Pi 5

This provides a pre‑configured virtual environment that includes all necessary libraries.

## Clone the Hailo Examples
```bash
# Clone the Hailo examples (includes HailoRT bindings, post‑processing, etc.)
git clone https://github.com/hailo-ai/hailo-rpi5-examples.git
cd hailo-rpi5-examples
```

## Run the Setup Script
This creates a virtual environment and installs all dependencies.
```bash
./setup_env.sh
```

## Activate the Environment
```bash
source setup_env.sh
```
*After activation, you can install any additional packages your scripts need (e.g., ultralytics, opencv-python, pyserial):*
```bash
pip install ultralytics opencv-python pyserial
```
✅ This method is tested and guarantees compatibility with the Hailo‑8 NPU on the Raspberry Pi 5.


### 4. Download the calibration and model files 
*(see the `calibration/` and `models/` folders).*

# 🔧 Calibration

Before running the measurement pipeline, you must generate the IE scale tables using the `multi_board_calib_zoomable.py` script.

```bash
python scripts/multi_board_calib_zoomable.py
```

Place a printed checkerboard at 1 m, 1.5 m, and 2 m from the camera.

Manually click the four outer corners of the board in each image.

The script produces `scale_table_w_ie_outside.npy` and `scale_table_h_ie_outside.npy`.

All calibration files are stored in the `calibration/` folder.

---

# 🧠 Model Training

The YOLOv8‑seg model was trained on a filtered dataset (6360/1809/918 train/val/test) for 60 epochs. The final metrics:

- **mAP50:** 0.916 (box), 0.913 (mask)
- **Precision:** 0.881
- **Recall:** 0.858

To retrain your own model, use the training script in the `docs/` folder (see ONNX conversion instructions). The trained weights are available in the `models/` folder.

# 🚀 Running the Measurement Pipeline

## Live camera (RPi Camera, upside‑down mounting)
```bash
python scripts/pothole_measure_hailo_v4.py \
    --input rpi \
    --hef models/pothole_seg_detector.hef \
    --no_flip \
    --save_video \
    --gps \
    --weather "Sunny"
```
*Use `--no_flip` if your camera is mounted upright.*

*Omit `--gps` if no GPS module is connected.*

## Pre-recorded video file
```bash
python scripts/pothole_measure_hailo_v4.py \
    --input path/to/video.mp4 \
    --hef models/pothole_seg_detector.hef \
    --save_video
```

## Output files
| File | Description |
| --- | --- |
| `pothole_detections_YYYYMMDD_HHMMSS.csv` | All detections with dimensions, repair boxes, GPS, weather |
| `pothole_annotated_YYYYMMDD_HHMMSS.mp4` | Annotated video (bounding boxes, repair boxes, text) |
| `pothole_crops_YYYYMMDD_HHMMSS/` | Cropped images of each tracked pothole |

## 🎥 Video Input & Output Examples
Example input and output videos are provided in the `outputs/` folder:
- **Sample input** – `outputs/sample_input.mp4` (raw camera feed)
- **Sample output** – `outputs/sample_output.mp4` (annotated with masks, boxes, and dimensions)

> 💡 The annotated video demonstrates the system’s performance on a real road scene.

## 📊 Results & Report
The final project report (PDF) is available in the repository root. Key quantitative results:

| Metric | Value |
| --- | --- |
| Detection mAP50 | 91.6% |
| Avg width error (IE) | 6.74% |
| Avg height error (IE) | 7.86% |
| Processing speed | ≈13 FPS (live, 1080p) |

For a full discussion, limitations, and future work, please read the final report.

# 📁 Repository Structure

**Automatic-Pothole-Detection-Classification-and-Sizing/**

```
├── scripts/               # All Python scripts (measurement, calibration, validation)
├── calibration/           # Camera intrinsic matrices and scale tables (.npy)
├── models/                # Trained YOLOv8‑seg model (.pt and .hef)
├── outputs/               # Sample input/output videos and example CSV
├── dashboard/             # Flask web dashboard (contains a .zip with all files)
├── docs/                  # Documentation (ONNX → HEF conversion guide, etc.)
├── LICENSE                # MIT License
└── README.md              # This file
```

## ⚖️ License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements
- Hailo Technologies for the Hailo‑8 accelerator and SDK
- Ultralytics for the YOLOv8 framework
- Roboflow for dataset hosting
- My supervisor, for guidance throughout the project

# 📬 Contact

**Lithemba Baninzi** – [GitHub](https://github.com/LithembaBaninzi)

**Project Link:** [https://github.com/LithembaBaninzi/Automatic-Pothole-Detection-Classification-and-Sizing](https://github.com/LithembaBaninzi/Automatic-Pothole-Detection-Classification-and-Sizing)
