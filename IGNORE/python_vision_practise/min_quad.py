import cv2
import numpy as np
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

    return mask

def find_minimum_bounding_quadrilateral(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to track the minimum bounding quadrilateral
    min_rect = None
    min_area = float('inf')

    # Iterate through the contours
    for contour in contours:
        # Find the minimum bounding rectangle for the contour
        rect = cv2.minAreaRect(contour)

        # Calculate the area of the bounding rectangle
        area = rect[1][0] * rect[1][1]

        # Update if the area is smaller than the previous minimum
        if area < min_area:
            min_rect = rect
            min_area = area

    return min_rect

# Sample usage:
# Load an image and apply a binary mask (mask) to it
image = cv2.imread("your_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

blue_mask = perform_blue_mask(image)

# Call the function to find the minimum bounding quadrilateral
min_bounding_rect = find_minimum_bounding_quadrilateral(blue_mask)

# Draw the minimum bounding quadrilateral on the original image
box = cv2.boxPoints(min_bounding_rect)
box = np.int0(box)
cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

# Display the result
cv2.imshow("Minimum Bounding Quadrilateral", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
