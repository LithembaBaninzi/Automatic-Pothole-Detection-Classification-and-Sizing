# This folder contains the steps needed for converting model weight from ONNX to Hef format

## Getting started
- Download the dataflow compiler from Hailo Developer Zone
- Download the Model Zoo from Hailo Developer Zone
- Create a folder in your Google Drive and upload the downloaded files
- Create a zip with your ONNX file and images to recalibrate the model on (best to use the validation images from your training, or you can use the training images)
- Also upload it to the Google Drive folder 
**Note:** Please make the files accessible to anyone with the link on the share option for all the files in the folder, as we will need them later

## System Requirements
To be able to run the dataflow compiler, you will need to have the following:
- Ubuntu 22.04 
- x86 Linux machine
- At least 16 GB RAM (32 GB Recommended)
You can use an AWS EC2 instance for this, but it's paid (not that expensive, it's around $0.5 per hour )

## Conversion 
On your Ubuntu machine (you can't run this in your Raspberry Pi), run the following commands: 
``` bash
sudo apt-get update

sudo apt install python3-virtualenv

# Verify if you are using Python 3.10
python3
# You should see Python 3.10 (Date and time) ... 
exit()

# Create a new virtual environment
virtualenv venv --python=python3

# Activate the virtual env
source venv/bin/activate

# Install gdown to download Google Drive files
pip install gdown

# Download the files - Run this command for each of the files in the folder
gdown fileID
# this is how you see the fileID on the link you copy 
# https://google.com/file/d/**fileID**/view?usp=drive_link

# Install the graphviz package needed by the dataflow compiler
sudo apt-get install -y graphviz graphviz-dev

# Install the dataflow compiler
pip install hailo_dataflow_compiler-3.33.1-py3-none-linux_x86_64.whl

# Install the Model Zoo
pip install hailo_model_zoo-2.18.0-py3-none-any.whl

# Install unzip if necessary
sudo apt install unzip 

# Unzip the data zip
unzip data.zip -d .

# Install required Python package
sudo apt-get install python3-tk

# Execute the model zoo
hailomz compile --help   # To check the correct syntax 
hailomz compile --ckpt ./data/filename.onnx --hw-arch hailo8 --classes 1 --calib-path ./data/pothole_images_cal/ --yaml ./venv/lib/python3.10/site-packages/hailo_model_zoo/cfg/networks/yolov8s.yaml
# The output will be saved as yolov8s.hef

# Copy the file into your host PC
scp -i "~/.ssh/aws-ft-key-pair.pem" ubuntu@ec2-52-91-95-109.compute-1.amazomaws.com:~yolov8s.hef ./pothole_detector.hef
```
Then shut down the EC2 instance or your Linux machine; you are done with it 

--- 
On your Raspberry Pi
``` bash
cd hailo-rpi5-examples

# Activate the virtual environment
source setup_env.sh

# Confirm if everything works with the pretrained model before testing your model
python basic_pipelines/detection.py --input rpi


# Copy the pothole_detector.hef file from your host pc
scp ./pothole_detector.hef pi@raspberrypi.local:~/ # you can use ip address instead of raspeberrypi.local

# Then check the file on your Raspberry Pi
ls ~/ #you should see pothole_detector.hef

# Run your custom model 
python basic_pipelines/detection.py --input rpi --heg-path ~/pothole_detector.hef
# You can ignore the person detection
```

## Common Errors and Fixes
If you get the following error while trying to run the *hailomz compile* command
``` bash
No such file or directory: '/home/ubuntu/venv/lib/python3.10/site-packages/hailo_model_zoo/cfg/alls/generic/../../postprocess_config/yolov8s_nms_config.json'
```
To fix this, you can follow these steps:
``` bash
# Clone the git repository
git clone https://github.com/hailo-ai/hailo_model_zoo.git

# Copy the postprocess directory 
cp -r hailo_model_zoo/hailo_model_zoo/cfg/postprocess_config /home/ubuntu/venv/lib/python3.10/site-packages/hailo_model_zoo/cfg/
```
Then rerun the compile command - It should work now
