import cv2
import numpy as np
def create_white_mask(image):

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary image
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # Invert the binary image (optional, if you want white elements to be white)
    inverted_binary_image = cv2.bitwise_not(binary_image)

    #Convert the input image to HSL
    cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    
    #White color mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)
    


    return white_mask # Return the white elements mask


# Load the image
image_path = '/Users/cianmurphy/code_directories/computer_vision/python_vision_ass/balls/Ball10.jpg'
image = cv2.imread(image_path)
cv2.imshow("Original", image)
black_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)


height = image.shape[0]
width = image.shape[1]

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow("gray image",gray)

# Apply Canny edge detection
edges = cv2.Canny(image, 150, 200, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

large_contours = []
for contour in contours:
    if cv2.contourArea(contour) > 50:  # Adjust the area threshold as needed
        large_contours.append(contour)

# Draw the four largest contours on the image
cv2.drawContours(black_image, large_contours, -1, (0, 255, 0), 3)

cv2.imshow("black_image",black_image)
# cv2.imshow("white mask", create_white_mask(image))




# Display the image
# cv2.imshow('Image with Four Largest Contours', black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
