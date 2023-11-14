
"""param1: This is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller). 
It's used in the edge detection stage. A higher value of param1 will lead to detecting fewer edges. This is because a 
higher threshold value will only allow the stronger edges to be detected (i.e., more likely to be part of a circle's 
circumference in this context).

param2: This parameter is the accumulator threshold for the circle centers at the detection stage. The smaller it is,
 the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first, 
 so if you're only looking for the most prominent circles, a higher value of param2 can be used. This is a threshold for 
 detecting the centers of the circles: the smaller the threshold, the more circles will be detected (including false circles).
   The larger the threshold, the more likely you are to only detect the prominent circles.

In essence, param1 is used by the edge detection process and param2 is used to decide when to successfully report a circle detected. 
Tuning these parameters is crucial for the successful detection of circles in an image, and they may require adjustment depending 
on the quality and characteristics of the input images."""

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


def part1(ball_num,show):
    path = f"balls/Ball{ball_num}.jpg"
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

    print("Image: ", ball_num, " => ",circle_info)

    if show : cv2.imshow("Detected Circles", img)

    if show: cv2.waitKey(0)
    if show: cv2.destroyAllWindows()

