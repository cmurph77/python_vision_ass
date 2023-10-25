import cv2
import numpy as np

# Draws 4 points on an image
def drawPoints(points, img):
    for i in range(len(points)):
        print("\n  - Circle:", i+1)

        # Draw a small circle (of radius 10) to show the center
        cv2.circle(img, (int(points[i][0]), int(points[i][1]), 10), (0, 255, 0), -1)

    # cv2.imshow("image", img)

def table_4_points():





# Code
