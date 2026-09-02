# YOLOv8 Webcam Tracking

Real-time object detection and tracking using YOLOv8 on Apple Silicon via a MacBook webcam. Runs on the PyTorch MPS (GPU) backend by default, with an optional Core ML export for Neural Engine inference. Includes persistent object IDs, trajectory trails, and a live FPS / latency overlay.

## Features

- Real-time object detection via YOLOv8 (defaults to `yolov8l.pt`)
- Multi-object tracking (ByteTrack) with persistent IDs across frames
- Trajectory trails showing recent movement path per tracked object
- Rolling FPS and inference-latency overlay for comparing backends
- Optional Core ML export for Apple Neural Engine inference

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

Run the tracking script (PyTorch MPS):

```bash
python webcam_yolo.py
```

Swap model size without editing the script:

```bash
python webcam_yolo.py --model yolov8n.pt
```

- A window will open showing your webcam feed with bounding boxes, class labels, tracking IDs, and motion trails.
- The overlay shows rolling end-to-end FPS, inference latency, backend (`mps` or `coreml`), and model name.
- Press `q` to quit. A short average FPS / inference summary is printed in the terminal.
- On first run, YOLOv8 will auto-download the weights file if it is not already present.

## Core ML export (Neural Engine)

Export converts a `.pt` checkpoint to a `.mlpackage`. Tracking stays in Python; only the detector runs through Core ML (typically on the Neural Engine).

```bash
python export_coreml.py --model yolov8l.pt
python webcam_yolo.py --model yolov8l.mlpackage
```

`coremltools` is installed automatically on first export. Hold model size and input resolution constant when comparing MPS vs Core ML — ignore the first ~10–30 frames after a Core ML load (compile / warmup), then use the overlay and the quit-time summary.

Fair comparison checklist: same `--model` variant (`l` vs `n`), same `imgsz` (export default 640), same camera, same trail drawing.

## Notes

- Tracking IDs are not permanent: if an object leaves the frame or is occluded for too long, ByteTrack may assign it a new ID upon reappearing rather than recognizing it as the same object.
- Trail length (how many past points are remembered per object) is controlled by the `TRAIL_LENGTH` constant in `webcam_yolo.py`.

## Roadmap / Ideas

- Counting line for unique object counts (e.g., traffic counting)
- Zone-based entry/exit alerts
