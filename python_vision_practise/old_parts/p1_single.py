
import cv2
import numpy as np


def is_color(hsv_color, lower, upper):
    mask = cv2.inRange(hsv_color, lower, upper)
    return cv2.countNonZero(mask) > 0


def is_color_orange(hsv_color):
    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([35, 255, 255])
    return is_color(hsv_color, orange_lower, orange_upper)


def is_color_white(hsv_color):
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 100, 255])
    return is_color(hsv_color, white_lower, white_upper)


def detect_color(img, x, y, r):
    center = (int(x), int(y)) # create a center object

    # Create a mask with the same dimensions as the input image but only one channel.
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Draw the circle on the mask with white color and filled (-1).
    cv2.circle(mask, center, int(r), (255), -1)

    # Now calculate the mean color using the mask. The mask must be 8-bit single-channel.
    mean_color = cv2.mean(img, mask=mask)

    mean_color_bgr = np.uint8([[mean_color[:3]]])
    mean_color_hsv = cv2.cvtColor(mean_color_bgr, cv2.COLOR_BGR2HSV)

    # print("mean color hsv:",mean_color_hsv, "for circle at x: ",x, ", y: ", y, ", r: ",r)
    return mean_color_hsv


path = f"balls/Ball2.jpg"
img = cv2.imread(path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_matrix = 5
gray_blurred = cv2.GaussianBlur(gray, (blur_matrix, blur_matrix), 0)

# Parameters for the Hough circles method
max_ball_radius = 200
min_ball_radius = 10
param_1 = 90        
param_2 = 40
min_dist = gray_blurred.shape[0] // 10  # Height/ 10

detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, minRadius=min_ball_radius,
                                    maxRadius=max_ball_radius, param1=param_1, param2=param_2, minDist=min_dist)

# Create a list to store the circle information
circle_info = []

# Draw detected circles and store their coordinates and radii
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for i in detected_circles[0, :]:
        x = i[0]
        y = i[1]
        center = (x, y)
        radius = i[2]
        color = detect_color(img, x, y, radius)
        if is_color_orange(color) or is_color_white(color):
            cv2.circle(img, center, radius, (255, 0, 0), 2)
            cv2.circle(img, center, 1, (0, 0, 255), 3)
            circle_info.append((center, radius))

print(circle_info)

cv2.imshow("Detected Circles", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
