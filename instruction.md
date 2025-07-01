# Kitchen Monitor Usage Guide

## Application Startup

### 1. Run Application
```bash
# Use default settings
python app-menu.py

# Specify custom video
python app-menu.py --video path/to/video.mp4

# Specify custom config
python app-menu.py --config path/to/config.yml

# Specify custom models
python app-menu.py --detection-model path/to/detect.pt --classification-model path/to/cls.pt
```

### 2. Application Interface
- **Video Area**: Display video with tracking and classification
- **Sidebar**: Display information and controls
- **Trackbar**: Adjust detection/classification parameters

## Application Controls

### Keyboard Shortcuts
| Key | Function |
|-----|----------|
| **ESC** | Exit application |
| **Z** | Enter zone drawing mode |
| **Left click** | Drag sliders, select options |
| **Right click** | Activate feedback for object |

### Trackbars
| Parameter | Description | Value |
|-----------|-------------|-------|
| **Confidence** | Detection confidence threshold | 0-100 |
| **IoU** | IoU threshold for detection | 0-100 |
| **ClassScore** | Class score threshold | 0-100 |
| **Objectness** | Objectness threshold | 0-100 |
| **ClsConf** | Classification confidence threshold | 0-100 |
| **FrameSkip** | Skip frames | 1-10 |
| **Mode** | Detection/classification mode | 0-1 |

## Drawing Zones

### Step 1: Activate Drawing Mode
1. Press **Z** key to enter zone drawing mode
2. Message displays: "Drawing READY zone, 4 points remaining..."

### Step 2: Draw Ready Zone (Red)
1. Click 4 points to draw **Ready Zone** (ready to serve area)
2. Zone is drawn in red (Amber)
3. Message: "Ready zone defined. Now click 4 points for PREPARE zone."

### Step 3: Draw Prepare Zone (Yellow)
1. Click 4 points to draw **Prepare Zone** (preparation area)
2. Zone is drawn in yellow (Teal)
3. Message: "Prepare zone defined. Zone definition complete!"

### Step 4: Complete
- Zone drawing mode automatically turns off
- Zones are saved and used for tracking

## State and Zone Logic

### Zone System
- **Prepare Zone (Yellow)**: Food preparation area
- **Ready Zone (Red)**: Ready to serve area

### State Transitions
```
Object appears → PREPARING (Teal)
Object enters Prepare Zone → PREPARING (Teal)
Object enters Ready Zone → READY_TO_SERVE (Green-Mint)
Object leaves Ready Zone → SERVED (Amber) [Event Trigger]
Object leaves all zones → REMOVED (Slate Blue)
```

### Classification Logic
- **Ready Classes**: `dish-not-empty`, `tray-not-empty`, `dish-kakigori`, `tray-kakigori`
  - Always `READY_TO_SERVE` when in Ready Zone
- **Empty Classes**: `dish-empty`, `tray-empty`
  - ≤ 5s: `READY_TO_SERVE`
  - > 5s: `PREPARING` + Alert
- **Unknown Classes**: Always `PREPARING`

### Bounding Box Colors
- **PREPARING**: Teal `[204, 182, 0]`
- **READY_TO_SERVE**: Green-Mint `[50, 205, 153]`
- **SERVED**: Amber `[13, 148, 252]`
- **REMOVED**: Slate Blue `[140, 90, 40]`
- **ALERT**: Red `[42, 42, 255]`

## Human-in-the-Loop Feedback System

### Activate Feedback
1. **Right-click** on incorrectly classified object
2. Feedback menu appears with question "What is the correct label?"
3. **Left-click** on correct label to select
4. Feedback automatically saved

### Feedback Data Structure
```
feedback_data/
├── dish-empty/
├── dish-not-empty/
├── tray-empty/
├── tray-not-empty/
├── dish-kakigori/
├── tray-kakigori/
└── feedback_log.csv
```

### Feedback Log Format
```csv
timestamp,image_path,model_prediction,user_correction
1703123456,feedback_data/dish-not-empty/1703123456_123.png,dish-empty,dish-not-empty
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