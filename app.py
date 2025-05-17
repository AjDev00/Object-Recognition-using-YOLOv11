import cv2
from flask import Flask, Response, render_template_string
from ultralytics import YOLO

app = Flask(__name__)

ip_cam_url = "http://10.108.224.5:8080/video"  # Replace with your IP webcam URL

cap = cv2.VideoCapture(ip_cam_url)

model = YOLO("yolo11n.pt")  # Your YOLOv11 model path

last_frame = None
frame_count = 0

def gen_frames():
    global last_frame, frame_count
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        if frame_count % 2 != 0 and last_frame is not None:
            # Yield last processed frame to skip detection for this frame
            ret, buffer = cv2.imencode('.jpg', last_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        # Resize frame to speed up detection
        small_frame = cv2.resize(frame, (320, 240))

        # Run detection
        results = model(small_frame)

        # Draw boxes on the resized frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item() * 100
                cls = int(box.cls[0].item())
                label = f"{model.names[cls]}: {conf:.1f}%"

                if conf < 80:  # Confidence threshold
                    continue

                cv2.rectangle(small_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(small_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        last_frame = small_frame.copy()

        # Encode with reduced JPEG quality for faster streaming
        ret, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template_string('''
    <html>
      <head>
        <title>YOLOv11 Live Detection</title>
        <style>
          body, html {
            margin: 0; padding: 0; overflow: hidden; background: black;
            height: 100%; width: 100%;
          }
          img {
            width: 100vw; height: 100vh; object-fit: contain;
            display: block;
            margin: 0 auto;
          }
        </style>
      </head>
      <body>
        <img src="{{ url_for('video_feed') }}" alt="Live Detection Stream">
      </body>
    </html>
    ''')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
