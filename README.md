# YOLOv8 Webcam Tracking

Real-time object detection and tracking using YOLOv8, running on Apple Silicon (MPS backend) via a MacBook webcam. Includes persistent object IDs and trajectory trail visualization.

## Features

- Real-time object detection via YOLOv8n
- Multi-object tracking (ByteTrack) with persistent IDs across frames
- Trajectory trails showing recent movement path per tracked object

## Requirements

- macOS with Apple Silicon (M1 or later)
- Python 3.9+
- A webcam

## Setup

1. Clone the repo and move into it:
```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
```

2. Create and activate a virtual environment:
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Usage

Run the tracking script:

```bash
python webcam_yolo.py
```

- A window will open showing your webcam feed with bounding boxes, class labels, tracking IDs, and motion trails.
- Press `q` to quit.
- On first run, YOLOv8 will auto-download the `yolov8n.pt` weights file — this only happens once.

## Notes

- Tracking IDs are not permanent: if an object leaves the frame or is occluded for too long, ByteTrack may assign it a new ID upon reappearing rather than recognizing it as the same object.
- Trail length (how many past points are remembered per object) is controlled by the `TRAIL_LENGTH` constant in `webcam_yolo.py`.

## Roadmap / Ideas

- Counting line for unique object counts (e.g., traffic counting)
- Core ML export for Neural Engine inference
- Zone-based entry/exit alerts
