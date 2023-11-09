import cv2
import numpy as np

def find_table_corners(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges using the Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours and find the one with 4 corners (a rectangle)
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            table_corners = approx
            break

    # Draw the detected rectangle on the original image
    cv2.drawContours(image, [table_corners], -1, (0, 255, 0), 2)

    # Extract and print the coordinates of the table corners
    for point in table_corners:
        x, y = point[0]
        print(f"Corner at ({x}, {y})")
        center = (x,y)
        cv2.circle(image, center, 5, (0, 0, 255), -1)


    # Display the image with the detected table corners
    cv2.imshow("Table Corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Specify the path to your image
image_path = "../Table4.jpg"

# Call the function to find and print the table corners
find_table_corners(image_path)
