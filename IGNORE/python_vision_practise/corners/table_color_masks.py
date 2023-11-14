# this attempt used color masks for the color white and blue

import cv2
import numpy as np

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

def perform_blue_mask(img):
    # Define the blue color range in HSV format
    blue_lower = np.array([100, 40, 30])
    blue_upper = np.array([130, 255, 255])

    # Convert the input image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for the blue color
    mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Perform dilations and erosions on the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Apply the mask to the original image
    blue_masked_image = cv2.bitwise_and(img, img, mask=mask)

    return blue_masked_image

def create_white_mask(image):

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary image
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # Invert the binary image (optional, if you want white elements to be white)
    inverted_binary_image = cv2.bitwise_not(binary_image)

    return binary_image  # Return the white elements mask



image_path = '../../tables/Table4.jpg'
img = cv2.imread(image_path)
blue_mask = perform_blue_mask(img)
white_mask  = create_white_mask(img)

cv2.imshow("blue_mask", blue_mask)
cv2.imshow("white_mask",white_mask)




cv2.waitKey(0)  
cv2.destroyAllWindows() 


