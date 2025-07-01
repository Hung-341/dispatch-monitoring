# Kitchen Dispatch Monitor - AI-Powered Object Tracking System

## Project Overview

Intelligent kitchen monitoring system using AI to detect, track, and classify the status of objects (dishes, trays) in video. The system helps control the time objects spend in danger zones and alerts when thresholds are exceeded.

### Key Features

- **Object Detection**: Detect dishes and trays using YOLOv8
- **Object Tracking**: Track objects across frames with custom DeepSORT implementation
- **Classification**: Classify status (empty, not-empty, kakigori)
- **Zone Management**: Manage preparation and ready zones
- **State Management**: Real-time object state management
- **Human-in-the-Loop**: Feedback system to improve models
- **Real-time Monitoring**: Real-time monitoring with visual interface

## Project Structure

```
dispatch-monitor/
├── app-menu.py                 # Main application with GUI
├── utils.py                    # Custom DeepSORT implementation
├── config.yml                  # System configuration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── README.md                   # This documentation
├── instruction.md              # Usage guide
├── .gitignore                  # Git ignore rules
├── .dockerignore               # Docker ignore rules
│
├── models/                     # Model directory
│   ├── best-detect.pt         # Detection model
│   └── best-cls.pt            # Classification model
│
├── videos/                     # Video input directory
│   └── videos.mp4             # Sample video
│
├── data/                       # Training data
│   ├── detection/             # Detection data
│   │   ├── images/
│   │   └── labels/
│   └── classification/        # Classification data
│       ├── dish-empty/
│       ├── dish-not-empty/
│       ├── tray-empty/
│       ├── tray-not-empty/
│       ├── dish-kakigori/
│       └── tray-kakigori/
│
├── feedback_data/              # HITL feedback data
│   ├── dish-empty/
│   ├── dish-not-empty/
│   ├── tray-empty/
│   ├── tray-not-empty/
│   ├── dish-kakigori/
│   ├── tray-kakigori/
│   └── feedback_log.csv       # Feedback log
│
├── runs/                       # YOLO training outputs
├── venv/                       # Virtual environment
└── __pycache__/                # Python cache
```

## Environment Setup

### Method 1: Local Installation

#### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
# Install required libraries
pip install -r requirements.txt
```

### Method 2: Using Docker

#### 1. Build Docker Image
```bash
docker build -t kitchen-monitor .
```

#### 2. Run with Docker Compose
```bash
docker-compose up --build
```

## Project Configuration

### 1. Training Data Structure

#### Detection Dataset
```
data/detection/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

**Label format (YOLO):**
```
class_id x_center y_center width height
```

#### Classification Dataset
```
data/classification/
├── dish-empty/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── dish-not-empty/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── tray-empty/
├── tray-not-empty/
├── dish-kakigori/
└── tray-kakigori/
```

## Object Tracking Implementation

### Custom DeepSORT Implementation

The project uses a **custom implementation** of the DeepSORT algorithm in `utils.py`:

#### Components:
- **KalmanBoxTracker**: Kalman Filter for bounding box tracking
- **Sort**: Main tracking class with Hungarian algorithm
- **IoU Matching**: Association based on Intersection over Union
- **State Management**: Track object states across frames

#### Features:
- **Real-time tracking**: Track objects across frames
- **ID assignment**: Assign unique ID to each object
- **Occlusion handling**: Handle when objects are occluded
- **Configurable parameters**: max_age, min_hits, iou_threshold

#### Usage:
```python
from utils import Sort

# Initialize tracker
tracker = Sort(max_age=60, min_hits=2, iou_threshold=0.2)

# Update with detections
tracks = tracker.update(detections)  # detections: [x1, y1, x2, y2, confidence]
```

### Alternative: BoxMOT (Optional)

If you want to use BoxMOT instead of custom implementation:

```bash
# Clone BoxMOT repository
git clone https://github.com/mikel-brostrom/boxmot.git

# Install BoxMOT
cd boxmot
pip install -e .

# Use in code
from boxmot import BotSort
tracker = BotSort()
```

## Training Models with YOLOv8

### 1. Training Detection Model

#### Prepare Data
```bash
# Create data.yaml for detection
cat > data/detection/data.yaml << EOF
path: ../data/detection
train: images/train
val: images/val

nc: 2  # number of classes
names: ['dish', 'tray']  # class names
EOF
```

#### Training
```bash
# Training with YOLOv8n
yolo train model=yolov8n.pt data=data/detection/data.yaml epochs=100 imgsz=640

# Or with custom config
yolo train model=yolov8n.pt data=data/detection/data.yaml epochs=100 imgsz=640 \
    batch=16 device=0 project=runs/detect name=detection_model
```

### 2. Training Classification Model

#### Prepare Data
```bash
# Create data.yaml for classification
cat > data/classification/data.yaml << EOF
path: ../data/classification
train: .  # train from current directory
val: .    # val from current directory

nc: 6  # number of classes
names: ['dish-empty', 'dish-not-empty', 'tray-empty', 'tray-not-empty', 'dish-kakigori', 'tray-kakigori']
EOF
```

#### Training
```bash
# Training classification model
yolo train model=yolov8n-cls.pt data=data/classification/data.yaml epochs=50 imgsz=224

# Or with custom config
yolo train model=yolov8n-cls.pt data=data/classification/data.yaml epochs=50 imgsz=224 \
    batch=32 device=0 project=runs/classify name=classification_model
```

### 3. Export Models
```bash
# Export detection model
yolo export model=runs/detect/detection_model/weights/best.pt format=torchscript

# Export classification model  
yolo export model=runs/classify/classification_model/weights/best.pt format=torchscript
```

## Application Usage

### 1. Run Application

#### Using Default Settings
```bash
python app-menu.py
```

#### Specify Custom Files
```bash
# Specify video
python app-menu.py --video path/to/video.mp4

# Specify config
python app-menu.py --config path/to/config.yml

# Specify models
python app-menu.py --detection-model path/to/detect.pt --classification-model path/to/cls.pt

# All options
python app-menu.py \
  --video path/to/video.mp4 \
  --config path/to/config.yml \
  --detection-model path/to/detect.pt \
  --classification-model path/to/cls.pt
```

### 2. Application Controls

#### Keyboard Shortcuts
- **ESC**: Exit application
- **Z**: Enter zone drawing mode
- **Left click**: Drag sliders, select options in feedback menu
- **Right click**: Activate feedback for object

#### Drawing Zones
1. Press **Z** to enter zone drawing mode
2. Click 4 points to draw **Ready Zone** (red)
3. Click 4 points to draw **Prepare Zone** (yellow)
4. Complete zone drawing

#### Adjust Parameters
- Use sliders in sidebar to adjust:
  - **Confidence**: Detection confidence threshold
  - **IoU**: IoU threshold for detection
  - **ClassScore**: Class score threshold
  - **Objectness**: Objectness threshold
  - **ClsConf**: Classification confidence threshold
  - **FrameSkip**: Skip frames
  - **Mode**: Detection/classification mode

## State and Zone Logic

### 1. Zone System

#### Prepare Zone (Yellow)
- **Purpose**: Food preparation area
- **Color**: Teal (Blue-green)
- **Logic**: Object in zone → `PREPARING` state

#### Ready Zone (Red)  
- **Purpose**: Ready to serve area
- **Color**: Amber (Yellow-orange)
- **Logic**: Object in zone → `READY_TO_SERVE` state
- **Alert**: > 30s → Red alert

### 2. State Management

#### State Transitions
```
Object appears → PREPARING (Teal)
Object enters Prepare Zone → PREPARING (Teal)
Object enters Ready Zone → READY_TO_SERVE (Green-Mint)
Object leaves Ready Zone → SERVED (Amber) [Event Trigger]
Object leaves all zones → REMOVED (Slate Blue)
```

#### Classification-Based Logic

**Ready Classes** (always READY_TO_SERVE):
- `dish-not-empty`, `tray-not-empty`
- `dish-kakigori`, `tray-kakigori`

**Empty Classes** (time limited):
- `dish-empty`, `tray-empty`
- ≤ 5s: READY_TO_SERVE
- > 5s: PREPARING + Alert

**Unknown Classes**:
- Always PREPARING

### 3. Visual Feedback

#### Bounding Box Colors
- **PREPARING**: Teal `[204, 182, 0]`
- **READY_TO_SERVE**: Green-Mint `[50, 205, 153]`
- **SERVED**: Amber `[13, 148, 252]`
- **REMOVED**: Slate Blue `[140, 90, 40]`
- **ALERT**: Red `[42, 42, 255]`

#### Label Format
```
ID: {track_id} | {state} | Status: {class_name}:{confidence}
```

## Human-in-the-Loop Feedback System

### 1. Activate Feedback

#### How to Use
1. **Right-click** on incorrectly classified object
2. **Select correct label** from popup menu
3. **Feedback automatically saved**

### 2. Feedback Data Structure

```
feedback_data/
├── dish-empty/
│   ├── 1703123456_123.png
│   └── 1703123457_124.png
├── dish-not-empty/
│   └── 1703123458_125.png
├── tray-empty/
├── tray-not-empty/
├── dish-kakigori/
├── tray-kakigori/
└── feedback_log.csv
```

#### Feedback Log Format
```csv
timestamp,image_path,model_prediction,user_correction
1703123456,feedback_data/dish-not-empty/1703123456_123.png,dish-empty,dish-not-empty
1703123457,feedback_data/dish-empty/1703123457_124.png,dish-not-empty,dish-empty
```

### 3. HITL Benefits

- **Model improvement**: Feedback data used for retraining
- **Easy to use**: Just right-click and select
- **Automated**: No manual operations required
- **Tracking**: Log file helps track improvement process

## Docker Usage

### Quick Start
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### Custom Configuration
```bash
# Mount custom config
docker run -v $(pwd)/config.yml:/app/config.yml \
           -v $(pwd)/models:/app/models \
           -v $(pwd)/videos:/app/videos \
           -v $(pwd)/feedback_data:/app/feedback_data \
           kitchen-monitor
```

## Package for Submission

### 1. Prepare Files
Ensure project directory contains:
- `app-menu.py`
- `utils.py`
   - `requirements.txt`
   - `README.md`
   - `config.yml`
   - `Dockerfile`
   - `docker-compose.yml`
   - `models/`
   - `videos/`
- `feedback_data/`

### 2. Create Package
```bash
# Windows
# Select all files/folders, right-click > Send to > Compressed (zipped) folder

# Linux/Mac
zip -r kitchen_monitor_poc.zip .
```

## Information Display

### Label Format
```
ID: {track_id} | {state} | Status: {class_name}:{confidence}
```

### Sidebar Information
- **READY ZONE OBJECTS**: List of objects in Ready Zone
- **Class**: Classified class name
- **State**: Current state
- **Dwell**: Time spent in zone (seconds)
- **Empty Dwell**: Time empty object spent in Ready Zone (if any)

## Troubleshooting

### Common Issues

#### 1. Cannot open video
- Check video path in `--video` argument
- Ensure video file exists and is readable

#### 2. Model cannot load
- Check model path in `--detection-model` and `--classification-model`
- Ensure model file (.pt) exists

#### 3. Feedback cannot save
- Check write permissions for `feedback_data/` directory
- Ensure sufficient disk space

#### 4. GUI not displaying
- On Linux: Install X11 forwarding
- On Windows: Use WSL2 or native Python

### Performance Tips
- Reduce **FrameSkip** to increase FPS
- Reduce **Confidence** to detect more objects
- Increase **IoU** to reduce false positives
- Use GPU if available

## Support

If you encounter issues, please:
1. Check logs in terminal
2. Ensure all dependencies are installed
3. Check configuration in `config.yml`
4. Contact team for support

## Contact

- **Author**: Hung-341
- **Email**: [hunglg.341@gmail.com]
- **Project**: Intelligent Kitchen Dispatch Monitor

---

**Note**: Ensure sufficient GPU memory when training models. With YOLOv8n, 4GB VRAM is sufficient for most cases.

---

## 4. Usage Guide

### Run Application
- **Local:**  
  `python app-menu.py`
- **Docker Compose:**  
  `docker-compose up --build`

### Web/GUI Interface
- Access via exposed port (default: `localhost:8080` if web interface is enabled).

### Managing Models & Data
- Place trained models in `models/`
- Input videos in `videos/`
- Feedback and training data in `feedback_data/` and `data/`

---

## 5. Technical Details

### Object Detection & Tracking
- **Detection:** YOLOv8
- **Tracking:** Custom DeepSORT in `utils.py`
- **Classification:** Status (empty, not-empty, kakigori)

### Zone & State Management
- Real-time management of preparation and ready zones
- Object state tracking across frames

### Optional: BoxMOT Integration
- See [BoxMOT](https://github.com/mikel-brostrom/boxmot) for advanced tracking

---

## 6. Model Improvement with User Feedback

- **Human-in-the-Loop (HITL):**  
  User feedback is collected and stored in `feedback_data/` for model retraining.
- **Feedback Logging:**  
  All feedback is logged in `feedback_log.csv`.

---

## 7. Model Training & Update

### Training Detection Model
- Prepare data as described above
- Use YOLOv8 for training:
```bash
# Example (adjust paths as needed)
yolo detect train data=data/detection/data.yaml model=yolov8n.pt
```

### Training Classification Model
- Organize images in `data/classification/`
- Use your preferred classification training pipeline

### Updating Models
- Place new models in `models/` and restart the service

---

## 8. Contribution & Development

### How to Contribute
- Fork the repository, create a feature branch, and submit a pull request.
- Please follow the code style and add clear commit messages.

### Issue Reporting
- Use GitHub Issues for bug reports and feature requests.

---

## 9. Contact & License

**Team:** Hung-341  
**Contact:** hunglg.341@gmail.com

---

*For more details, please refer to the comments and docstrings in the source code.*
