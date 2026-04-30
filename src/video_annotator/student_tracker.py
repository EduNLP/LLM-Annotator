import cv2
import random
import subprocess
from ultralytics import YOLO
import os
import numpy as np
from deepface import DeepFace
# STEP 1: Import the necessary modules.
import mediapipe as mp


yolo = YOLO("yolov8x.pt")
detector = "yolov8x"
align = True
vid      = "my_movie.mp4"
videoCap = cv2.VideoCapture(vid)

width  = int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = videoCap.get(cv2.CAP_PROP_FPS)

temp_out  = "my_movie.avi"
final_out = "my_movie_annotated_detected.mp4"
fourcc    = cv2.VideoWriter_fourcc(*"MJPG")
out       = cv2.VideoWriter(temp_out, fourcc, fps, (width, height))



acceptable = ["person", "book", "laptop", "notebook", "paper", "pen", "tablet", "ipad"]


def getColours(cls_num):
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

def load_gallery(save_dir="histograms/"):
    if not os.path.exists(save_dir):
        print(f"Warning: '{save_dir}' not found")
        return {}
    gallery = {}
    for file in os.listdir(save_dir):
        if file.endswith(".npy"):
            name = os.path.splitext(file)[0]
            gallery[name] = np.load(os.path.join(save_dir, file))
    print(f"Loaded gallery: {list(gallery.keys())}")
    return gallery

def match_gallery(hist, gallery, threshold=0.6):
    if not gallery:
        return "Unknown", 0.0
    best_name, best_score = "Unknown", -1
    for name, ref_hist in gallery.items():
        score = cv2.compareHist(
            hist.reshape(8, 8, 8).astype(np.float32),
            ref_hist.reshape(8, 8, 8).astype(np.float32),
            cv2.HISTCMP_CORREL
        )
        if score > best_score:
            best_score = score
            best_name  = name
    return (best_name, best_score) if best_score >= threshold else ("Unknown", best_score)

# ✅ Load gallery ONCE
gallery = load_gallery()

# ✅ Track ID → name map (persists across frames)
id_to_name = {}

frame_count = 0

while True:
    ret, frame = videoCap.read()
    if not ret:
        break

    results = yolo.track(frame, stream=True, persist=True)  # persist=True is key

    for result in results:
        class_names = result.names

        for box in result.boxes:
            if box.conf[0] < 0.4:
                continue

            # ✅ Skip if no track ID assigned yet
            if box.id is None:
                continue

            track_id   = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls        = int(box.cls[0])
            class_name = class_names[cls]

            if class_name not in acceptable:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if track_id not in id_to_name:
                hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                                    [0, 180, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                name, score = match_gallery(hist, gallery)
                id_to_name[track_id] = name
                print(f"New track {track_id} → '{name}' (score: {score:.3f})")
            else:
                name = id_to_name[track_id]

            conf   = float(box.conf[0])
            colour = getColours(track_id)   # color tied to track_id, not class

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"{name} id:{track_id} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            
    out.write(frame)
    frame_count += 1

videoCap.release()
out.release()
print(f"Frames written: {frame_count}")
print(f"Final ID map: {id_to_name}")



subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_out,
    "-vcodec", "mpeg4",
    "-q:v", "2",
   final_out
], check=True)


os.remove(temp_out)
print(f"Saved: {final_out}")