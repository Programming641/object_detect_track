from ultralytics import YOLO
import cv2, os, shutil, json
import numpy as np

model = YOLO("yolov8s-seg.pt")

vid_name = "cat2"
cap = cv2.VideoCapture(f"./{vid_name}.mp4")

output_dir = f"output/{vid_name}"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

frame_index = 0
track_obj = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.3)
    r = results[0]

    track_obj[frame_index] = {}

    if r.boxes is not None and r.masks is not None and r.boxes.id is not None:
        ids = r.boxes.id.cpu().numpy()
        polygons = r.masks.xy  # ✅ already scaled to image size

        for tid, poly in zip(ids, polygons):
            tid = int(tid)

            # polygon pixels in IMAGE coordinates
            pixels = poly.astype(int).tolist()

            track_obj[frame_index][tid] = pixels

    cv2.imwrite(f"{output_dir}/{frame_index}.png", frame)
    frame_index += 1

    if frame_index == 50:
        break

cap.release()

with open(f"output/{vid_name}.json", "w") as f:
    json.dump(track_obj, f)




























