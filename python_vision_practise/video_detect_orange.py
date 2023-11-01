import cv2
import numpy as np
from numpy import True_
# import part1

# opt 1= orange_mask, 2= just mask
def perform_orange_mask(img,opt):
    # Define the blue color range in HSV format
    orange_lower = np.array([15, 100, 100])
    orange_upper = np.array([30, 255, 255])

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



def video_analysis():
    # Open the video file
    video_path = "TableTennis.avi"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("Error: Could not open the video file."); exit();

    frame_count = 0;
    analyse = True;
    while analyse:
        frame_count = frame_count + 1
        if(frame_count >100): analyse = False; # end after 100 frames
        print("FRAME_COUNT: ",frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Perform frame analysis here
        analyse_frame(frame,frame_count)

        # Display the frame
        # cv2.imshow('Frame', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def analyse_frame(frame, count):
        # Display the frame
        cv2.imshow('Frame', frame)
        orange_mask = perform_orange_mask(frame,1)
        cv2.imshow("mask",orange_mask)


video_analysis()
cv2.waitKey(0)
cv2.destroyAllWindows()
