from ultralytics import YOLO
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a YOLO checkpoint to Core ML for Neural Engine inference"
    )
    parser.add_argument(
        "--model",
        default="yolov8l.pt",
        help="Path to PyTorch weights to export",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export input size (static; ANE prefers this)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    exported = model.export(
        format="coreml",
        nms=True,
        quantize=16,
        imgsz=args.imgsz,
    )
    print(f"Exported Core ML model: {exported}")
    print(f"Run it with: python webcam_yolo.py --model {exported}")


if __name__ == "__main__":
    main()
