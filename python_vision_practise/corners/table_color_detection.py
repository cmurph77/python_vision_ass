import cv2
import numpy as np

# Load the image
image_path = "../Table4.jpg"
image = cv2.imread(image_path)
cv2.imshow("original image",image)

# Convert the image from BGR to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define a range of blue color in HSV
lower_blue = np.array([90, 40, 30])  # Lower bound for blue in HSV
upper_blue = np.array([140, 255, 255])  # Upper bound for blue in HSV

# Create a binary mask for blue colors
mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Bitwise-AND the original image and the mask to get the blue regions
blue_masked_image = cv2.bitwise_and(image, image, mask=mask)


# Display the original image and the mask
# cv2.imshow('Original Image', image)
cv2.imshow('Blue Mask', mask)
# cv2.imshow('Blue Masked Image', blue_masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
