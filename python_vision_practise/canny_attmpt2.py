import cv2
import numpy as np

# Load the image
image = cv2.imread('tables/Table1.jpg')

# Preprocessing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Edge detection with Canny
edges = cv2.Canny(gray, 50, 150)  # You can adjust these threshold values

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter and approximate contours
filtered_contours = []
for contour in contours:
    if len(contour) >= 5:  # Filter based on the number of vertices
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            filtered_contours.append(approx)

# Extract and filter corners
table_corners = []
for contour in filtered_contours:
    for point in contour:
        table_corners.append(tuple(point[0]))

# Visualize the corners
for corner in table_corners:
    cv2.circle(image, corner, 5, (0, 0, 255), -1)


# Display the result
cv2.imshow('Table Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
