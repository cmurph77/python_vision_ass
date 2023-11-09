# this works on all but picture 10 and doesnt detect all 3 balls in picture 5
""""

Image No:  1
BALL DETECTED
      [((562, 310), 41)]
 
Image No:  2
BALL DETECTED
      [((432, 458), 68)]
 
Image No:  3
BALL DETECTED
      [((416, 406), 47)]
 
Image No:  4
BALL DETECTED
      [((364, 382), 57)]
 
Image No:  5
BALL DETECTED
      [((440, 362), 42)]
 
Image No:  6
BALL DETECTED
      [((384, 330), 48)]
 
Image No:  7
BALL DETECTED
      [((528, 282), 36)]
 
Image No:  8
BALL DETECTED
      [((524, 458), 30)]
 
Image No:  9
BALL DETECTED
      [((530, 404), 30)]
 
Image No:  10
BALL DETECTED
BALL DETECTED
BALL DETECTED
      [((90, 44), 41), ((140, 32), 31), ((186, 32), 24)]"""
import cv2
import numpy as np

#  TODO get color detection working


def is_color(hsv_color, lower, upper):
    mask = cv2.inRange(hsv_color, lower, upper)
    return cv2.countNonZero(mask) > 0


def is_color_orange(hsv_color):
    orange_lower = np.array([15, 100, 100])
    orange_upper = np.array([35, 255, 255])
    return is_color(hsv_color, orange_lower, orange_upper)


def is_color_white(hsv_color):
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 100, 255])
    return is_color(hsv_color, white_lower, white_upper)


def detect_color(img, x, y, r):
    center = (int(x), int(y))
    cv2.circle(img, center, int(r), (0, 255, 0), 2)
    cv2.circle(img, center, 1, (0, 0, 255), 3)

    # Create a mask with the same dimensions as the input image but only one channel.
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Draw the circle on the mask with white color and filled (-1).
    cv2.circle(mask, center, int(r), (255), -1)


    # You can display the mask if you want to see it.
    # cv2.imshow("color mask", mask)

    # Now calculate the mean color using the mask. The mask must be 8-bit single-channel.
    mean_color = cv2.mean(img, mask=mask)

    mean_color_bgr = np.uint8([[mean_color[:3]]])
    mean_color_hsv = cv2.cvtColor(mean_color_bgr, cv2.COLOR_BGR2HSV)

    # print("mean color hsv:",mean_color_hsv, "for circle at x: ",x, ", y: ", y, ", r: ",r)
    return mean_color_hsv


def get_ball_center_coords(img, show):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_matrix = 5
    gray_blurred = cv2.GaussianBlur(gray, (blur_matrix, blur_matrix), 0)

    # Parameters for the Hough circles method
    max_ball_radius = 200
    min_ball_radius = 10
    # param_1 = 150
    # param_2 = 65
    param_1 = 90
    param_2 = 40
    min_dist = gray_blurred.shape[0] // 10  # Height/8

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
            # if is_color_orange(color): print("orange")
            # if is_color_white(color): print("white")
            if is_color_orange(color) or is_color_white(color):
                # print("Color is orange or white")
                # cv2.circle(img, center, radius, (0, 255, 0), 2)
                # cv2.circle(img, center, 1, (0, 0, 255), 3)
                circle_info.append((center, radius))


    if (show):
        cv2.imshow("Detected Circles", img)

    return circle_info


def find_center_handler(image_no):
    path = f"balls/Ball{image_no}.jpg"
    ball_img = cv2.imread(path)
    cv2.imshow("Oringinal Image", ball_img)
    print("Image No: ", image_no)
    center = get_ball_center_coords(ball_img, 1)
    print("     ", center)


def iterate_through_ball_images():
    for i in range(1, 11):
        print(" ")
        find_center_handler(i)


#
iterate_through_ball_images()
# find_center_handler(10)

cv2.waitKey(0)
cv2.destroyAllWindows()
