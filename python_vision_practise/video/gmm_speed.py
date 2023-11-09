import cv2
import numpy as np
import time

# Create a VideoCapture object to read from a video file or camera
video_source = '../TableTennis.avi'  # Replace with your video source
cap = cv2.VideoCapture(video_source)

# Initialize the background subtractor with GMM
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Initialize time variables
prev_time = time.time()
current_time = time.time()
 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)
    cv2.imshow("fg mask", fg_mask)
    
    # Post-process the mask (optional)
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # Find contours in the mask to detect moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the speed of each detected object
    current_time = time.time()
    time_diff = current_time - prev_time
    prev_time = current_time

    # Draw rectangles around fast-moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            object_speed = w / time_diff  # Speed is calculated as change in width per second
            if object_speed < 1000:  # Adjust the speed threshold (in pixels per second) as needed
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow('Fast Moving Object Detection', frame)

    # Display the original frame and the result
    cv2.imshow('Original Video', frame)
    og_masked_image = cv2.bitwise_and(frame, frame, mask=fg_mask)
    cv2.imshow("og masked image",og_masked_image)

    cv2.imshow('Foreground Mask', fg_mask)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
