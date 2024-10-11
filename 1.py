import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import BooleanVar

# Function to load the model, advertisement image, and video dynamically
def load_model():
    global model
    model_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("YOLO Model Files", "*.pt")])
    if model_path:
        model = YOLO(model_path)
        model_label.config(text=f"Selected Model: {model_path.split('/')[-1]}")

def load_ad_image():
    global ad_img
    ad_img_path = filedialog.askopenfilename(title="Select Advertisement Image", filetypes=[("Image Files", "*.jpg *.png")])
    if ad_img_path:
        ad_img = cv2.imread(ad_img_path)
        ad_label.config(text=f"Selected Ad Image: {ad_img_path.split('/')[-1]}")

def load_video():
    global video_path
    video_path = filedialog.askopenfilename(title="Select Billboard Video", filetypes=[("Video Files", "*.mp4 *.avi")])
    if video_path:
        video_label.config(text=f"Selected Video: {video_path.split('/')[-1]}")

# Function to start the video processing
def run_program():
    if model is None or ad_img is None or not video_path:
        messagebox.showerror("Error", "Please select a model, advertisement image, and video before running the program.")
        return
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open video.")
        return

    # Setup video writer if saving is enabled
    if save_video.get():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter('output_with_ad.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading video.")
            break

        # Run YOLOv8 detection on the current frame
        results = model(frame)

        # Process each detected object (assuming billboard is class 0)
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Adjust class ID based on your setup
                    x_min, y_min, x_max, y_max = box.xyxy[0]
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                    # Resize the advertisement image to fit the detected billboard area
                    ad_resized = cv2.resize(ad_img, (x_max - x_min, y_max - y_min))

                    # Define the corners for perspective transformation
                    billboard_corners = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype="float32")
                    ad_corners = np.array([[0, 0], [ad_resized.shape[1], 0], [ad_resized.shape[1], ad_resized.shape[0]], [0, ad_resized.shape[0]]], dtype="float32")

                    # Perform perspective transformation
                    matrix = cv2.getPerspectiveTransform(ad_corners, billboard_corners)
                    warped_ad = cv2.warpPerspective(ad_resized, matrix, (frame.shape[1], frame.shape[0]))

                    # Create a mask for overlay
                    mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillConvexPoly(mask, billboard_corners.astype(int), (255, 255, 255))

                    # Overlay the advertisement on the detected billboard
                    frame = cv2.bitwise_and(frame, cv2.bitwise_not(mask))
                    frame = cv2.add(frame, warped_ad)

        # Display the frame with the ad overlay
        cv2.imshow('Billboard Ad Replacement', frame)

        # Write to output video if enabled
        if save_video.get():
            output.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_video.get():
        output.release()
    cv2.destroyAllWindows()

# Create the main application window
root = tk.Tk()
root.title("Billboard Ad Replacement")

# Variables for UI
model = None
ad_img = None
video_path = None
save_video = BooleanVar()

# UI elements
model_button = tk.Button(root, text="Select YOLO Model", command=load_model)
model_button.pack(pady=10)

model_label = tk.Label(root, text="No Model Selected")
model_label.pack()

ad_button = tk.Button(root, text="Select Ad Image", command=load_ad_image)
ad_button.pack(pady=10)

ad_label = tk.Label(root, text="No Ad Image Selected")
ad_label.pack()

video_button = tk.Button(root, text="Select Billboard Video", command=load_video)
video_button.pack(pady=10)

video_label = tk.Label(root, text="No Video Selected")
video_label.pack()

save_checkbutton = tk.Checkbutton(root, text="Save Output Video", variable=save_video)
save_checkbutton.pack(pady=10)

run_button = tk.Button(root, text="Run Program", command=run_program)
run_button.pack(pady=20)

# Start the main loop
root.mainloop()
