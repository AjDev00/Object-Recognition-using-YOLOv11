import cv2
import tkinter as tk
from PIL import Image, ImageTk
from detect import detect_objects

# Use DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set lower resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Function to update the camera feed
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = detect_objects(frame)  # Apply YOLOv11 detection
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.flip(frame, 1)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
    lbl_video.after(10, update_frame)

# Initialize Tkinter window
root = tk.Tk()
root.title("YOLOv11 Object Detection")

# Label to display video
lbl_video = tk.Label(root)
lbl_video.pack()

# Start updating the frame
update_frame()

# Run the application
root.mainloop()

# Release the camera
cap.release()
cv2.destroyAllWindows()
