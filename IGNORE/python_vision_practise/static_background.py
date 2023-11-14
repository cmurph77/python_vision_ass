# this is taken from kens slides and converted to python
# absdiff( current_frame, first_frame, difference );
# cvtColor( difference, moving_points, CV_BGR2GRAY );
# threshold( moving_points, moving_points, 30, 255, THRESH_BINARY );
# Mat display_image = Mat::zeros( moving_points.size(), CV_8UC3 );
# current_frame.copyTo( display_image, moving_points );
# 
# Taken from video slides 

import cv2
import numpy as np

# opt 1= orange_mask, 2= just mask
def perform_orange_mask(img,opt):
    # Define the blue color range in HSV format
    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([35, 255, 255])

    # Convert the input image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for the blue color
    mask = cv2.inRange(hsv, orange_lower, orange_upper)

    # Perform dilations and erosions on the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Apply the mask to the original image
    orange_masked_image = cv2.bitwise_and(img, img, mask=mask)

    if(opt == 1) : return orange_masked_image;
    if(opt == 2) : return mask;


# Open a video capture object
video_capture = cv2.VideoCapture('TableTennis.avi')  # Replace 'your_video_file.mp4' with your video file path

# Read the first frame to initialize
ret, first_frame = video_capture.read()
if not ret:
    raise Exception("Error reading the first frame")

while True:
    
    ret, current_frame = video_capture.read()
    
    if not ret:
        break  # End of the video

    # Calculate the absolute difference between the current frame and the first frame
    difference = cv2.absdiff(current_frame, first_frame)

    # Convert the difference to grayscale
    moving_points = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary mask of moving objects
    _, moving_points = cv2.threshold(moving_points, 30, 255, cv2.THRESH_BINARY)

    # Create an output image filled with zeros
    display_image = np.zeros_like(current_frame)

    # Overlay the current frame with moving objects
    display_image = cv2.bitwise_and(current_frame, current_frame, mask=moving_points)

    orange_mask = perform_orange_mask(display_image,1)
    cv2.imshow("mask",orange_mask)

    # Display the current frame with moving objects highlighted
    # cv2.imshow('Moving Objects', display_image)
    # cv2.waitKey(0)
    # Press 'q' to exit the video analysis
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
