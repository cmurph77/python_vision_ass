import cv2
import numpy as np
import time

# Initialize the camera or read a video file
video_source = '../TableTennis.avi'  # Replace with your video source
cap = cv2.VideoCapture(video_source)

# Read the first frame to create the initial background
ret, background = cap.read()

# Convert the background to grayscale for background subtraction
background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

# Initialize time variables
prev_time = time.time()
current_time = time.time()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the absolute difference between the background and the current frame
    frame_diff = cv2.absdiff(background_gray, frame_gray)
    
    # Apply a threshold to create a binary mask of the moving objects
    _, thresholded = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate the speed of each detected object
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    # Draw rectangles around fast-moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            object_speed = w / time_diff  # Speed is calculated as change in width per second
            if object_speed < 500:  # Adjust the speed threshold (in pixels per second) as needed
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow('Fast Moving Object Detection', frame)
    
    if cv2.waitKey(30) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
