import cv2
from ultralytics import YOLO  # Ensure ultralytics is compatible with YOLOv11

# Load YOLOv11 model (Ensure "yolov11n.pt" is in the correct directory)
model = YOLO("yolo11n.pt")  # Change path if the file is in another location

# Object detection function
def detect_objects(frame):
    results = model(frame)  # Run YOLOv11 inference on the frame
    height, width, _ = frame.shape
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item() * 100  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = f"{model.names[cls]}: {conf:.2f}"  # Class label + confidence
            
            # Draw bounding box and label (do not apply flip to text)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame  # Return frame with detections
