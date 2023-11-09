import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Define a function to calculate the center of the ball
def find_ball_center(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)

    # Detect the circles in the image
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    # If at least one circle is detected
    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # Draw the circle in the output image, and a rectangle corresponding to the center of the circle
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        # Return the center of the first circle detected
        return (circles[0][0], circles[0][1])
    else:
        return None

# Path to the directory containing the images
image_dir_path = 'balls/'

# List all the image files in the directory
image_files = [file for file in os.listdir(image_dir_path) if file.startswith('Ball')]

# Initialize a dictionary to store the centers
ball_centers = {}

# Process each image file
for image_file in image_files:
    # Find the ball center
    center = find_ball_center(os.path.join(image_dir_path, image_file))
    # Store the center in the dictionary
    ball_centers[image_file] = center

ball_centers
