import cv2
import numpy as np
def create_white_mask(image):

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary image
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # Invert the binary image (optional, if you want white elements to be white)
    inverted_binary_image = cv2.bitwise_not(binary_image)

    return binary_image  # Return the white elements mask




# Load the image
image_path = 'Tables/Table2.jpg'
image = cv2.imread(image_path)
black_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)


white_mask = create_white_mask(image)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray image",gray)

# Apply Canny edge detection
edges = cv2.Canny(image, 150, 200, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the four largest contours on the image
cv2.drawContours(black_image, contours, -1, (0, 255, 0), 3)

height = image.shape[0]
width = image.shape[1]




# Display the image
cv2.imshow('Image with Four Largest Contours', black_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
