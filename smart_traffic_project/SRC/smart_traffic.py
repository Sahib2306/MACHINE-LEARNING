import cv2
import os
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# -----------------------------
# PATH CONFIG (matches your folder!)
# -----------------------------

MODEL_PATH = "../MODELS/helmet_detector.pt"       # model folder
VIDEO_PATH = "../DATA/input_video.mp4"            # video folder
OUTPUT_FOLDER = "../OUTPUT/violations"            # output images
LOG_PATH = "../OUTPUT/violations_log.csv"         # CSV log file

STOP_LINE_Y = 350             # adjust based on video
TRAFFIC_LIGHT = "RED"         # static for now

# Make sure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create log CSV if not present
if not os.path.exists(LOG_PATH):
    df = pd.DataFrame(columns=["time", "violation_type", "image_path"])
    df.to_csv(LOG_PATH, index=False)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (boxAArea + boxBArea - interArea)

def save_violation(violation_type, frame):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    filename = f"{violation_type}_{timestamp}.jpg"
    filepath = f"{OUTPUT_FOLDER}/{filename}"

    cv2.imwrite(filepath, frame)

    df = pd.read_csv(LOG_PATH)
    df.loc[len(df)] = [timestamp, violation_type, filepath]
    df.to_csv(LOG_PATH, index=False)

    return filepath

# -----------------------------
# LOGIC FUNCTIONS
# -----------------------------

def detect_helmet_violations(detections):
    bikes = [d for d in detections if d["class"] in ["motorbike", "motorcycle", "bike"]]
    persons = [d for d in detections if d["class"] == "person"]
    helmets = [d for d in detections if d["class"] in ["helmet", "with_helmet"]]

    violations = []

    for bike in bikes:
        rider = None

        for p in persons:
            if iou(bike["bbox"], p["bbox"]) > 0.2:
                rider = p
                break

        if rider is None:
            continue

        rx1, ry1, rx2, ry2 = rider["bbox"]
        head_zone = (rx1, ry1, rx2, ry1 + int(0.4 * (ry2 - ry1)))

        has_helmet = any(iou(head_zone, h["bbox"]) > 0.15 for h in helmets)

        if not has_helmet:
            violations.append({"type": "helmet", "bbox": bike["bbox"]})

    return violations

def detect_signal_violations(detections, stop_line_y):
    if TRAFFIC_LIGHT != "RED":
        return []

    vehicles = [d for d in detections if d["class"] in 
                ["car", "truck", "bus", "motorbike", "bike"]]

    violations = []

    for v in vehicles:
        x1, y1, x2, y2 = v["bbox"]
        if y2 < stop_line_y:
            violations.append({"type": "signal", "bbox": (x1, y1, x2, y2)})

    return violations

# -----------------------------
# MAIN PROGRAM
# -----------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

print("🚦 Smart Traffic System Running...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Video Finished")
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        classname = model.names[cls]

        detections.append({"class": classname, "bbox": (x1, y1, x2, y2)})

    # Helmet Violations
    for v in detect_helmet_violations(detections):
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,0,255), 3)
        cv2.putText(annotated, "NO HELMET", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        save_violation("helmet", annotated)

    # Signal Violations
    for v in detect_signal_violations(detections, STOP_LINE_Y):
        x1, y1, x2, y2 = v["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255,255,0), 3)
        cv2.putText(annotated, "SIGNAL VIOLATION", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        save_violation("signal", annotated)

    # Draw Stop Line
    cv2.line(annotated, (0, STOP_LINE_Y), (annotated.shape[1], STOP_LINE_Y),
             (255,0,0), 3)

    # -----------------------------
    # RESIZE OUTPUT WINDOW (IMPORTANT)
    # -----------------------------
    annotated = cv2.resize(annotated, (960, 540))   # Change size if needed

    cv2.imshow("Smart Traffic Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✔ Done. Check /OUTPUT/violations/")
