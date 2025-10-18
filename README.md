# AI-Powered Library Seat Monitoring System

An intelligent computer vision system that automatically monitors library seat occupancy and student engagement using YOLOv8 and MediaPipe.

## Features

- 🪑 **Automatic Seat Detection**: Dynamically detects and maps all chairs
- 🧍‍♂️ **Occupancy Detection**: Real-time tracking of empty/occupied seats
- 🧠 **Posture Analysis**: Detects focused vs distracted behavior
- 🔔 **Smart Reminders**: Gentle alerts for inactivity
- 📊 **Analytics Dashboard**: Real-time utilization metrics and heatmaps

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## System Requirements

- Python 3.8+
- GPU recommended for optimal performance
- Camera or video feed input
- 4GB+ RAM recommended

## Performance Targets

- Detection FPS: ≥ 15 FPS
- Occupancy Accuracy: ≥ 90%
- Max Persons per Camera: 40-50
- False Slack Alerts: ≤ 10%
