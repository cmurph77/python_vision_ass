import cv2
import numpy as np
def get_hist(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hue, saturation, value = cv2.split(orange_ball_hsv)
    # Histogram 
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist

video_source = '../TableTennis.avi'  # Replace with your video source
cap = cv2.VideoCapture(video_source)

# Read the first frame to select the object to track
ret, frame = cap.read()
if not ret:
    exit(1)

# Define the region of interest (ROI) around the object to track


# Calculate the histogram of the ROI
orange_ball = cv2.imread("../../balls/orange_ball.jpg")
roi_hist = get_hist(orange_ball)

# Normalize the histogram
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Set the termination criteria for the tracking
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the current frame to HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calculate the back projection of the current frame using the ROI histogram
    dst = cv2.calcBackProject([frame_hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)

    # Apply meanshift to find the new position of the object
    ret, track_window = cv2.meanShift(dst, (x, y, w, h), term_criteria)

    # Draw a rectangle around the tracked object
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with the tracking rectangle
    cv2.imshow('Object Tracking', frame)
    
    if cv2.waitKey(30) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
