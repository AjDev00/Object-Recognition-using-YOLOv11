import cv2
import tkinter as tk
from PIL import Image, ImageTk
from detect import detect_objects
from gtts import gTTS
import tempfile
import uuid
import os
import threading

cap = cv2.VideoCapture("http://10.108.224.5:8080/video")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_announced = set()

def play_audio_async(filename):
    # This function runs in a separate thread to avoid blocking
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

def update_frame():
    ret, frame = cap.read()
    if ret:
        frame, labels = detect_objects(frame)
        if labels:
            speak_objects(labels)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
    lbl_video.after(2, update_frame)  # ~50 FPS update rate (adjust if needed)

root = tk.Tk()
root.title("YOLOv11 Object Detection")

lbl_video = tk.Label(root)
lbl_video.pack()

update_frame()

root.mainloop()

cap.release()
cv2.destroyAllWindows()
