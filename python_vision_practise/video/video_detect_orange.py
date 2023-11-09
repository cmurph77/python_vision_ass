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
    video_path = "../TableTennis.avi"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print("Error: Could not open the video file."); exit();

    frame_count = 0;
    analyse = True;


    roi = cv2.imread("pitch_ground.jpg")
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    hue, saturation, value = cv2.split(hsv_roi)

    # Histogram ROI
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])

    while analyse:
        frame_count = frame_count + 1
        if(frame_count >100): analyse = False; # end after 100 frames
        print("FRAME_COUNT: ",frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        hsv_original = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # Perform frame analysis here
        mask = cv2.calcBackProject([hsv_original], [0, 1], roi_hist, [0, 180, 0, 256], 1)

        # Filtering remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # create a kernel for the 2d filter
        mask = cv2.filter2D(mask, -1, kernel)
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)  # removes more noise

        mask = cv2.merge((mask, mask, mask))  # create 3 channel mask for doing bitwise operations
        result = cv2.bitwise_and(frame, mask)

        cv2.imshow("Mask", mask)
        cv2.imshow("Original image", frame)
        cv2.imshow("Result", result)

        # Display the frame
        # cv2.imshow('Frame', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def analyse_frame(frame, count):
        # Display the frame
        # cv2.imshow('Frame', frame)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



video_analysis()
cv2.waitKey(0)
cv2.destroyAllWindows()
