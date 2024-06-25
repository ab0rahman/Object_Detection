# Object Detection using YOLOv9
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ab0rahman/Object_Detection/blob/main/Object_Detection.ipynb)

## About YOLO Algorithm:
**YOLO (You Only Look Once)** is a real-time object detection algorithm that operates as a single-stage object detector using a **convolutional neural network (CNN)** to predict bounding boxes and class probabilities for objects within input images. The algorithm divides the input image into a grid of cells, where each cell predicts the probability of the presence of an object and the bounding box coordinates of that object, as well as the object's class. Unlike two-stage object detectors such as R-CNN and its variants, YOLO processes the entire image in one pass, making it faster and more efficient. Over time, YOLO has been enhanced with features that improve accuracy, speed up processing, and handle small objects more effectively. Due to these improvements, YOLO is widely used for real-time object detection tasks, such as real-time video analytics and real-time video surveillance. The YOLO algorithm involves several key steps: grid division, bounding box prediction, and class prediction.

## Setting Up
We are utilizing Google Colab for its GPU resources to enhance computational efficiency. Additionally, you may use your local disk depending on your computer's capability.

**[Open Colab](https://colab.research.google.com/notebooks/intro.ipynb)**

**Change Runtime Type:**

1.Click on the "Runtime" menu at the top of the Colab interface.

2.Select "Change runtime type" from the dropdown menu Select GPU.

3.In the runtime type dialog choose the GPU type by selecting `Tesla T4` from the dropdown menu.

**Verify if a GPU is allocated**

Run `!nvidia-smi` in a Google Colab notebook to check the GPU specifications look if you're connected to a Tesla T4 GPU.

```bash
!nvidia-smi
```

```bash
Tue Jun 25 10:00:00 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   40C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
This output provides information about the GPU name (Tesla T4), driver version, CUDA version, GPU temperature, power usage, and memory usage. If you see this output after running !nvidia-smi, it confirms that your Colab notebook is connected to a T4 GPU.




## Clone the YOLOv9 Repository:

```bash
!git clone https://github.com/WongKinYiu/yolov9.git
```

Cloning a repository in Git allows you to obtain a full local copy of a project, enabling collaboration, version control, and development in isolated environments.

## Download Yolov9-e Model 

```bash
!wget  https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt
```
Use the `!wget` command to download yolov9-e.pt model from GitHub

YOLO models available

**1.YOLOv9-T**

**2.YOLOv9-S**

**3.YOLOv9-M**

**4.YOLOv9-C**

**5.YOLOv9-E**

YOLOv9-E is a variant of YOLOv9 optimized for efficiency in object detection tasks. The -E Variant in `yolov9-e.pt` typically indicates a specific configuration or variant of the YOLOv9 model. It could imply optimizations or adjustments made to the model architecture or training process to enhance efficiency or adapt it for specific use cases.


##  Change the working directory to yolov9

```bash
%cd yolov9
```

## Install Python packages 

```bash
!pip install -r requirements.txt -q
```
Install Python packages listed in `requirements.txt`
 
Few necessary packages are:

`scipy`
`PyTorch`
`matplotlib`
`seaborn`
`gitpython`
`opencv-python`

## Download a file from Google Drive

```bash
!gdown "https://drive.google.com/uc?id=1K4DhZcaXiS5md4qy2D1BQqyHMkScxENB"
```

```bash
!gdown "https://drive.google.com/uc?id=175E1FXTyyitxRp9A6PFiaqKIr4q8i2kY"
```

Download the File using `gdown` followed by the Google Drive file ID (id parameter from the sharing link)

**Obtain the File ID** of the file you want to download from Google Drive. You can find this ID in the sharing link of the file.

For example, if the sharing link is `https://drive.google.com/file/d/1K4DhZcaXiS5md4qy2D1BQqyHMkScxENB/view`, then `1K4DhZcaXiS5md4qy2D1BQqyHMkScxENB` is the file ID.

**Change Sharing Settings** 
In the sharing settings, change the setting to ***"Anyone with the link"***.


## Inference on Image and Videos using model 

```bash
!python detect_dual.py --weights '/content/yolov9-e.pt' --source 'image1.jpg' --device 0
```

```bash
!python detect_dual.py --weights '/content/yolov9-e.pt' --source 'movie3.mp4' --device 0
```
`detect_dual.py` performs object detection using the YOLOv9-E (Yolov9-Efficient) model to detect objects in images or videos.

`--weights` Path to the pre-trained weights file (yolov9-e.pt) for the YOLOv9-E model.

`--source` Input source for detection, such as an image (image1.jpg) or a video file.

`--device` Argument to specify the device (GPU) index to use for inference (0 in this case) no need in case of a (CPU).

The results will be saved to `runs/detect/exp`


## Display the Image

```bash
from IPython.display import Image

Image(filename='runs/detect/exp/image1.jpg')
```
Use the Image class from IPython.display

<p float="left">
  <img src="https://github.com/ab0rahman/Object_Detection/blob/main/results/image2.jpg?raw=true" width="500" />
  <img src="https://github.com/ab0rahman/Object_Detection/blob/main/results/image1.jpg?raw=true" width="400" /> 
</p>

## Display the Video

```bash
from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = 'runs/detect/exp2/movie3.mp4'

# Compressed video path
compressed_path = 'result_compressed.mp4'

# Compress the video using ffmpeg
os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Read the compressed video
with open(compressed_path, 'rb') as f:
    mp4 = f.read()

# Encode the video in base64
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

# Display the video
HTML(f"""
<video width=400 controls>
      <source src="{data_url}" type="video/mp4">
</video>
""")
```
Import the `HTML` class from IPython's display module, which allows embedding HTML content.

Import the `b64encode` function from the base64 module, which will be used to encode the video file.

**Video Compression**

`os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")` Uses ffmpeg a command-line tool for handling multimedia files to compress the input video using the `libx264` video codec the compressed video is saved at compressed path.

**Video Read and convert to HTML**

`mp4 = f.read()` Reads the content of the compressed video file into the variable `mp4`.

`data_url = "data:video/mp4;base64," + b64encode(mp4).decode()` Converts the binary content of mp4 into a base64-encoded string. This is necessary because HTML5 video tags can use base64-encoded data URIs to display video content directly

**Display the video**

This HTML string is passed to `HTML()`, which renders it as embedded video content within the  environment.
<p float="left">
  <img src="https://github.com/ab0rahman/Object_Detection/blob/main/results/movie2.gif" width="600" height="300"><br>
  <img src="https://github.com/ab0rahman/Object_Detection/blob/main/results/movie2%20(1).gif?raw=true" width="300" height="600"><br>
</p>

## Results

https://github.com/ab0rahman/Object_Detection/assets/143890577/ee8ab770-f12a-4bdb-9300-4122811b0d10 

https://github.com/ab0rahman/Object_Detection/assets/143890577/b0a594f4-d661-4872-9ac8-f7d2f45bb11e

# Collaborators
## 🔗 Links

  **Abdur Rahman**
  
[![github](https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ab0rahman)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdur-rahman-5491a824b/)
[![gmail](https://img.shields.io/badge/gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:letsmail.him@gmail.com)

