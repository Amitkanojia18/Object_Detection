import cv2
import os
import numpy as np
import torch  # for running the model, deep learning for PD

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLOv5 model
# Load the model and move it to GPU if available
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

# Function to detect and label objects in a frame
def detect_and_label(frame):
    resized_frame = cv2.resize(frame, (640, 360))

    results = model(resized_frame)

    # Extract boundary boxes, labels, confidence score to predict
    labels, confidences, boxes = results.xyxyn[0][:, -1], results.xyxyn[0][:, -2], results.xyxyn[0][:, :-2]

    # Define a scaling factor to shrink the bounding boxes
    scaling_factor = 0.8  # 80% of the original size

    # Converting boundary boxes and labels into understandable format
    for box, label, confidence in zip(boxes, labels, confidences):
        if confidence > 0.5:  # threshold
            x1, y1, x2, y2 = int(box[0] * frame.shape[1]), int(box[1] * frame.shape[0]), int(
                box[2] * frame.shape[1]), int(box[3] * frame.shape[0])
            class_name = model.names[int(label)]

            # Apply scaling factor
            width = x2 - x1
            height = y2 - y1
            x1 = int(x1 + (width * (1 - scaling_factor)) / 2)
            y1 = int(y1 + (height * (1 - scaling_factor)) / 2)
            x2 = int(x2 - (width * (1 - scaling_factor)) / 2)
            y2 = int(y2 - (height * (1 - scaling_factor)) / 2)

            # Classifying whether the person is an adult or child
            if class_name == 'person':
                if (y2 - y1) < frame.shape[0] / 2:
                    label = 'Child'
                else:
                    label = 'Adult'

                # Draw a smaller rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 0), 2)

    return frame

# Path to the folder where videos are stored
VIDEO_DIR = 'video/'

# Get the video file name from the user
video_name = input("Enter the full name of the video file (e.g., video1.mp4): ")

# Construct the full path to the video
video_path = os.path.join(VIDEO_DIR, video_name)

# Check if the file exists
if not os.path.exists(video_path):
    print(f"Error: {video_name} not found in {VIDEO_DIR}")
    exit(1)

# Load the video
video_capture = cv2.VideoCapture(video_path)

# Process the video frame by frame
while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    # Detect and label the frame
    labeled_frame = detect_and_label(frame)


    # Display the video with detections
    cv2.imshow('Video', labeled_frame)

    # Press 'q' to quit the video early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
video_capture.release()
cv2.destroyAllWindows()
