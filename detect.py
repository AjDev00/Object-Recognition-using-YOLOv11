import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

def detect_objects(frame, conf_threshold=0.8):
    results = model(frame)
    height, width, _ = frame.shape
    
    labels = []
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf < conf_threshold:
                continue  # Skip low confidence
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())
            label = f"{model.names[cls]}: {conf * 100:.2f}%"
            labels.append(model.names[cls])

            # Draw box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame, labels
