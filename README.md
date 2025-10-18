# AI-Powered Library Seat Monitoring System

An intelligent computer vision system that automatically monitors library seat occupancy and student engagement using YOLOv8 and computer vision techniques.

## Features

- 🪑 **Automatic Seat Detection**: Dynamically detects and maps all chairs
- 🧍‍♂️ **Occupancy Detection**: Real-time tracking of empty/occupied seats
- 🧠 **Posture Analysis**: Detects focused vs distracted behavior
- 🔔 **Smart Reminders**: Gentle alerts for inactivity
- 📊 **Analytics Dashboard**: Real-time utilization metrics and heatmaps
- 📈 **Data Logging**: Comprehensive analytics and reporting
- 🎯 **Performance Monitoring**: Real-time FPS and accuracy tracking

## Quick Start

1. **Create and activate virtual environment with Python 3.9:**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the system:**
   ```bash
   python test_system.py
   ```

4. **Run the monitoring system:**
   ```bash
   python main.py
   ```

## Installation

### Manual Installation

```bash
# Create virtual environment with Python 3.9
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (automatic on first run)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create directories
mkdir -p logs exports screenshots heatmaps
```

### System Requirements

- Python 3.9 (recommended for MediaPipe compatibility)
- GPU recommended for optimal performance
- Camera or video feed input
- 4GB+ RAM recommended
- OpenCV compatible camera
- Virtual environment support

**Note**: Python 3.9 is recommended as MediaPipe has compatibility issues with Python 3.13.

## Usage

### Basic Usage

```bash
# Run with default settings
python main.py

# Run with custom camera
python main.py --camera 1

# Run without dashboard
python main.py --no-dashboard

# Run without alerts
python main.py --no-alerts

# Run without data logging
python main.py --no-logging
```

### Advanced Usage

```bash
# Custom confidence threshold
python main.py --confidence 0.7

# Custom IoU threshold
python main.py --iou 0.4

# Use custom YOLO model
python main.py --model path/to/model.pt
```

### Keyboard Controls

- `q` or `ESC`: Quit the system
- `s`: Save screenshot
- `h`: Save heatmap
- `e`: Export data to CSV

## Configuration

The system can be configured through `config.py` or by creating a `config.json` file:

```json
{
  "detection": {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.3
  },
  "posture": {
    "inactivity_threshold": 30.0,
    "head_tilt_threshold": 30.0
  },
  "alert": {
    "enable_audio": true,
    "alert_cooldown": 60.0
  }
}
```

## Performance Targets

- Detection FPS: ≥ 15 FPS
- Occupancy Accuracy: ≥ 90%
- Max Persons per Camera: 40-50
- False Slack Alerts: ≤ 10%
- Processing Time: ≤ 200ms per frame

## Architecture

```
[Camera Input] → [YOLO Detection] → [Seat Association] → [Posture Analysis] → [Alert System] → [Visualization] → [Data Logging]
```

## File Structure

```
library productivity/
├── main.py                 # Main application
├── detection.py           # YOLO detection system
├── posture_analysis.py    # Posture analysis system
├── alert_system.py       # Alert and notification system
├── visualization.py       # Real-time visualization
├── data_logging.py       # Data logging and analytics
├── config.py             # Configuration management
├── test_system.py        # System testing
├── setup.py              # Setup script
├── requirements.txt      # Dependencies
├── yolov8n.pt           # YOLO model weights
├── venv/                # Virtual environment
├── logs/                # Log files
├── exports/             # Data exports
├── screenshots/         # Screenshots
├── heatmaps/           # Heatmap images
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check camera permissions and try different camera indices
2. **Low FPS**: Reduce frame resolution or disable dashboard
3. **Detection errors**: Adjust confidence threshold
4. **Audio alerts not working**: Check pygame installation
5. **Import errors**: Make sure virtual environment is activated
6. **MediaPipe issues**: The system uses alternative posture analysis methods
7. **Permission errors**: Run with appropriate camera permissions

### Testing

Run the test suite to verify system functionality:

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run tests
python test_system.py
```

### Virtual Environment Management

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Deactivate virtual environment
deactivate

# Remove virtual environment (if needed)
rm -rf venv
```

### Logs and Data

- Logs are stored in the `logs/` directory
- Database: `library_monitoring.db`
- Exports: `exports/` directory
- Screenshots: `screenshots/` directory
- Heatmaps: `heatmaps/` directory

## Privacy and Ethics

- No face recognition or personal identification
- Focuses only on posture and occupancy
- All data is anonymized
- Respects user privacy and study environment
