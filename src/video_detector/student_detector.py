from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import os

model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'])  # add 'de' for German, etc.

def save_histogram(hist, name):
    name = name.strip()
    name = name.replace(" ", "_")
    file_path = os.path.join("histograms/",f"{name}.npy")
    np.save(file_path, hist)
    print(f"saved::::  {file_path}")


def detect_objects(image_path, conf_threshold=0.5):
    # Load image
    image = cv2.imread(image_path)
    
    # Run inference
    results = model(image, conf=conf_threshold)[0]
    
    # Parse results
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = model.names[cls]
        print(label)
        if label != "person":
            continue
        
        # Crop the detected region
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

   


        # Run OCR on the crop
        ocr_results = reader.readtext(crop)


        # Extract text lines
        texts = [text for (_, text, conf) in ocr_results if conf > 0.3]
        full_text = " ".join(texts)

        
        # Color histogram (HSV)
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        save_histogram(hist, full_text)

        print(f"[{full_text}] histogram shape: {hist.shape}, sum: {hist.sum():.2f}")

        print(f"[{label}] → '{full_text}'")
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        text = f"{label} {conf:.2f}"
        cv2.putText(image, full_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        print(f"Detected: {label:15s} | Conf: {conf:.2f} | Box: [{x1},{y1},{x2},{y2}]")
    
    cv2.imwrite("output.jpg", image)
    return results

results = detect_objects("annotated.png")