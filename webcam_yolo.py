from ultralytics import YOLO
from collections import defaultdict, deque
from pathlib import Path
import argparse
import time
import cv2

TRAIL_LENGTH = 30
METRIC_WINDOW = 30


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 webcam tracking")
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Path to model weights (.pt or .mlpackage)",
    )
    return parser.parse_args()


def is_coreml_model(model_path):
    suffix = Path(model_path).suffix.lower()
    return suffix in {".mlpackage", ".mlmodel"}


def backend_label(model_path):
    return "coreml" if is_coreml_model(model_path) else "mps"


def draw_metrics(frame, fps, infer_ms, backend, model_name):
    text = f"FPS {fps:.1f} | infer {infer_ms:.0f} ms | {backend} | {model_name}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 6
    x, y = 8, 8
    cv2.rectangle(
        frame,
        (x, y),
        (x + text_w + pad * 2, y + text_h + baseline + pad * 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + pad, y + pad + text_h),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()
    model_path = args.model
    model_name = Path(model_path).stem
    backend = backend_label(model_path)
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)

    track_kwargs = {"persist": True, "verbose": False}
    if not is_coreml_model(model_path):
        track_kwargs["device"] = "mps"

    # track_id -> deque of (x, y) center points
    track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    loop_times = deque(maxlen=METRIC_WINDOW)
    infer_times = deque(maxlen=METRIC_WINDOW)

    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        infer_start = time.perf_counter()
        results = model.track(frame, **track_kwargs)
        infer_times.append((time.perf_counter() - infer_start) * 1000)

        annotated = results[0].plot()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                center = (float(x), float(y))

                # Add this frame's center point to that track's history
                track_history[track_id].append(center)

                # Draw the trail: connect consecutive points for this ID
                points = track_history[track_id]
                for i in range(1, len(points)):
                    pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    cv2.line(annotated, pt1, pt2, (0, 255, 255), 2)

        avg_infer = sum(infer_times) / len(infer_times)
        avg_loop = sum(loop_times) / len(loop_times) if loop_times else 0
        fps = (1.0 / avg_loop) if avg_loop > 0 else 0.0
        draw_metrics(annotated, fps, avg_infer, backend, model_name)

        cv2.imshow("YOLOv8 Tracking + Trails", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        loop_times.append(time.perf_counter() - loop_start)

    if loop_times:
        avg_loop = sum(loop_times) / len(loop_times)
        avg_infer = sum(infer_times) / len(infer_times)
        print(
            f"avg FPS {1.0 / avg_loop:.1f} | avg infer {avg_infer:.0f} ms | "
            f"{backend} | {model_name} ({len(loop_times)} frames)"
        )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
