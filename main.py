import cv2
import tkinter as tk
from PIL import Image, ImageTk
from detect import detect_objects
from gtts import gTTS
import tempfile
import uuid
import os
import threading
import time

# Use your IP camera
cap = cv2.VideoCapture("http://10.108.224.5:8080/video")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Shared variables
current_frame = None
processed_frame = None
labels_to_speak = []
frame_lock = threading.Lock()

last_announced = set()

def play_audio_async(filename):
    import playsound
    playsound.playsound(filename)
    try:
        os.remove(filename)
    except:
        pass

def speak_objects(labels):
    global last_announced
    new_labels = set(labels) - last_announced
    if new_labels:
        text = ", ".join(new_labels)
        tts = gTTS(text)
        temp_filename = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        tts.save(temp_filename)
        threading.Thread(target=play_audio_async, args=(temp_filename,), daemon=True).start()
        last_announced = set(labels)

def camera_worker():
    global current_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame.copy()
        time.sleep(0.01)

def detection_worker():
    global processed_frame, labels_to_speak
    while True:
        with frame_lock:
            frame = current_frame.copy() if current_frame is not None else None
        if frame is not None:
            frame, labels = detect_objects(frame)
            processed_frame = frame
            labels_to_speak = labels
        time.sleep(0.03)

def update_gui():
    global processed_frame
    if processed_frame is not None:
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
    lbl_video.after(10, update_gui)

def speaker_worker():
    global labels_to_speak
    while True:
        if labels_to_speak:
            speak_objects(labels_to_speak)
        time.sleep(2)

# Start threads
threading.Thread(target=camera_worker, daemon=True).start()
threading.Thread(target=detection_worker, daemon=True).start()
threading.Thread(target=speaker_worker, daemon=True).start()

# Tkinter GUI
root = tk.Tk()
root.title("YOLOv11 Object Detection")

lbl_video = tk.Label(root)
lbl_video.pack()

update_gui()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
