import cv2
import cvzone
import time
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "fall.mp4"
CLASSES_FILE = "classes.txt"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FRAME_SKIP = 2                  # process every nth frame
PROCESSING_SIZE = (416, 416)    # YOLO input size (smaller -> faster)
CONFIDENCE_THRESHOLD = 0.7

# Tracker params (point-based with Euclidean distance)
DISTANCE_THRESH = 75.0          # maximum pixel distance to match detections to tracks
HIT_COUNTER_MAX = 20
INITIALIZATION_DELAY = 2

# -------------------------
# MODEL + TRACKER INIT
# -------------------------
model = YOLO(MODEL_PATH)

tracker = Tracker(
    distance_function="euclidean",      # use euclidean on center points (not IoU)
    distance_threshold=DISTANCE_THRESH,
    hit_counter_max=HIT_COUNTER_MAX,
    initialization_delay=INITIALIZATION_DELAY
)

cap = cv2.VideoCapture(VIDEO_PATH)

with open(CLASSES_FILE, "r") as f:
    classnames = f.read().splitlines()

prev_time = 0
frame_count = 0

print("Starting Fall Detection with Norfair (point-based tracking)...")

# -------------------------
# MAIN LOOP
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to fixed size for display and stable tracking metrics
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_count += 1

    # Skip frames for speed
    if frame_count % FRAME_SKIP != 0:
        continue

    proc = cv2.resize(frame, PROCESSING_SIZE)

    # Run YOLO (stream=True yields results iterator)
    results = model(proc, stream=True, classes=[0], verbose=False, conf=CONFIDENCE_THRESHOLD)

    # FPS calc
    now = time.time()
    fps = 1.0 / (now - prev_time) if prev_time else 0.0
    prev_time = now

    # scale factors from processing to display frame
    sx = FRAME_WIDTH / PROCESSING_SIZE[0]
    sy = FRAME_HEIGHT / PROCESSING_SIZE[1]

    detections = []
    # Convert YOLO detections -> Norfair point detections
    for info in results:
        for box in info.boxes:
            x1_t, y1_t, x2_t, y2_t = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if cls == 0 and conf >= CONFIDENCE_THRESHOLD:
                x1 = float(x1_t.item()) * sx
                y1 = float(y1_t.item()) * sy
                x2 = float(x2_t.item()) * sx
                y2 = float(y2_t.item()) * sy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                detections.append(
                    Detection(
                        points=np.array([[cx, cy]]),
                        scores=np.array([conf]),
                        data=np.array([x1, y1, x2, y2], dtype=float)
                    )
                )

    # Update tracker with the point detections
    tracked_objects = tracker.update(detections=detections)

    # Find the newest tracked object (highest ID) if there are any
    newest_object = None
    if tracked_objects:
        newest_object = max(tracked_objects, key=lambda obj: obj.id)

    # Only process and draw the newest object
    objects_to_process = []
    if newest_object:
        objects_to_process.append(newest_object)

    # Draw status
    cvzone.putTextRect(frame, f'FPS: {int(fps)}', [20, 40], thickness=2, scale=2)
    cvzone.putTextRect(frame, f'Detections: {len(detections)}', [20, 80], thickness=2, scale=1)
    cvzone.putTextRect(frame, f'Total Tracked: {len(tracked_objects)}', [20, 110], thickness=2, scale=1)
    if newest_object:
        cvzone.putTextRect(frame, f'Processing ID: {newest_object.id}', [20, 140], thickness=2, scale=1, colorR=(0,255,0))

    # Process tracked objects (now only contains the newest one, if any)
    for obj in objects_to_process:
        if getattr(obj, "last_detection", None) is not None and getattr(obj.last_detection, "data", None) is not None:
            bbox = obj.last_detection.data
        else:
            est = obj.estimate
            cx, cy = float(est[0][0]), float(est[0][1])
            w = 60
            h = 160
            x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
            bbox = np.array([x1, y1, x2, y2], dtype=float)

        x1, y1, x2, y2 = map(int, bbox.tolist())
        w = x2 - x1
        h = y2 - y1
        ratio = h / (w + 1e-6)

        cvzone.cornerRect(frame, (x1, y1, w, h), l=30, rt=6)

        # =================================================================================
        # START: MODIFICATION FOR YOUR REQUEST
        # Set the label to just "human" without the ID
        # =================================================================================
        label = "human"  # MODIFICATION: Always label as "human"
        color = (0, 255, 0)

        # Simple fall detection by aspect ratio
        if ratio < 0.8:
            label = "FALL - human"  # MODIFICATION: Fall label for human
            color = (0, 0, 255)

            # optional: confirm by checking vertical movement
            if len(obj.past_detections) >= 3:
                current_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
                past_det = obj.past_detections[-3]
                if getattr(past_det, "points", None) is not None:
                    px = float(past_det.points[0][0])
                    py = float(past_det.points[0][1])
                    past_center = np.array([px, py])
                    vertical_movement = current_center[1] - past_center[1]
                    if vertical_movement > 20:
                        label = "FALL CONFIRMED - human"  # MODIFICATION
                        color = (0, 0, 255)
        # =================================================================================
        # END: MODIFICATION FOR YOUR REQUEST
        # =================================================================================

        cvzone.putTextRect(frame, label, [x1 + 8, max(12, y1 - 12)], thickness=2, scale=1.2, colorR=color)

    cv2.imshow("Fall Detection with Stable Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")