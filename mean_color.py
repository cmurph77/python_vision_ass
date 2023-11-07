import cv2
import numpy as np

def detect_color(img, x, y, r):
    center = (int(x), int(y))
    cv2.circle(img, center, int(r), (0, 255, 0), 2)
    cv2.circle(img, center, 1, (0, 0, 255), 3)

    mask = np.zeros_like(img)
    cv2.circle(mask, center, int(r), (255, 255, 255), -1)

    cv2.imshow("color mask",mask)

    # mean_color = cv2.mean(img, mask)

    # mean_color_bgr = np.uint8([mean_color[:3]])
    # mean_color_hsv = cv2.cvtColor(mean_color_bgr, cv2.COLOR_BGR2HSV)

    # return mean_color_hsv

# Load your image
img = cv2.imread("balls/Ball1.jpg")

# Define the center (x, y) and radius (r)
x = 564
y = 311
r = 40.5

# Call the function to detect the mean color in the specified circular region
mean_color_hsv = detect_color(img, x, y, r)

print("Mean Color (H, S, V):", mean_color_hsv)