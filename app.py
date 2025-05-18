import cv2
from ultralytics import YOLO

# Load YOLOv11 model (ensure yolov11n.pt or another model file is in your working directory)
model = YOLO("yolo11n.pt")

# Use your phone's IP Webcam stream URL here
cap = cv2.VideoCapture("http://10.108.224.5:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO object detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item() * 100  # Confidence
            cls = int(box.cls[0].item())  # Class ID
            label = f"{model.names[cls]}: {conf:.1f}%"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Object Detection Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
