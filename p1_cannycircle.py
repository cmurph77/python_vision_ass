import cv2
import numpy as np

# Load the image
image = cv2.imread('balls/Ball7.jpg')  # Replace 'your_image.jpg' with the path to your image

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
edges_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
cv2.drawContours(edges_image, contours, -1, (255, 0, 0), 3)
cv2.imshow("edges",edges_image)

 # Parameters for the Hough circles method
max_ball_radius = 200
min_ball_radius = 10
param_1 = 50
param_2 = 30
min_dist = image.shape[0] // 10  # Height/8
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minRadius=min_ball_radius, maxRadius=max_ball_radius, param1=param_1, param2=param_2, minDist=min_dist)

# If circles are found, draw them on the original image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(image, center, radius, (0, 255, 0), 2)  # Draw the circle
        cv2.circle(image, center, 2, (0, 0, 255), 3)  # Draw the center of the circle

# Display the resulting image
cv2.imshow('Circles Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
