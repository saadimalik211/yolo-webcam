from ultralytics import YOLO
from collections import defaultdict, deque
import cv2

model = YOLO("yolov8l.pt")
cap = cv2.VideoCapture(0)

# How many past points to remember per track ID
TRAIL_LENGTH = 30

# track_id -> deque of (x, y) center points
track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, device="mps", persist=True, verbose=False)
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

    cv2.imshow("YOLOv8 Tracking + Trails", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
