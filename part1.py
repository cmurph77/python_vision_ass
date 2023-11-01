import cv2
import numpy as np

#  TODO get color detection working

def test_function():
    print("Hello from part1.py")

def is_color(hsv_color, lower, upper):
    mask = cv2.inRange(hsv_color, lower, upper)
    return cv2.countNonZero(mask) > 0

def is_color_orange(hsv_color):
    orange_lower = np.array([15, 100, 100])
    orange_upper = np.array([30, 255, 255])
    return is_color(hsv_color, orange_lower, orange_upper)

def is_color_white(hsv_color):
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 100, 255])
    return is_color(hsv_color, white_lower, white_upper)

def detect_color(img, x, y, r):
    center = (int(x), int(y))
    cv2.circle(img, center, int(r), (0, 255, 0), 2)
    cv2.circle(img, center, 1, (0, 0, 255), 3)

    mask = np.zeros_like(img)
    cv2.circle(mask, center, int(r), (255, 255, 255), -1)

    mean_color = cv2.mean(img, mask)

    mean_color_bgr = np.uint8([mean_color[:3]])
    mean_color_hsv = cv2.cvtColor(mean_color_bgr, cv2.COLOR_BGR2HSV)

    if is_color_orange(mean_color_hsv):
        pass  # Your logic here

    return mean_color_hsv

def find_ball_center(img):
    # convert image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_matrix = 3
    gray_blurred = cv2.GaussianBlur(gray, (blur_matrix, blur_matrix), 0)
    
    # parameters for the hough circles method
    max_ball_radius = 100
    min_ball_radius = 10
    param_1 = 150
    param_2 = 70
    min_dist = gray_blurred.shape[0] // 20 # height/8
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1,minRadius=min_ball_radius, maxRadius=max_ball_radius,param1=param_1,param2=param_2,minDist=min_dist)

    # draw detected circles
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        count =1;
        for i in detected_circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img, center, radius, (0, 255, 0), 2)
            cv2.circle(img, center, 1, (0, 0, 255), 3)
            print(" -> circle: coords = (", i[0], ",", i[1], ") , r = ", radius, ")")
            count = count +1;

    cv2.imshow("Detected Circles", img)

def get_ball_center_coords(img, min_r, max_r,show):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_matrix = 3
    gray_blurred = cv2.GaussianBlur(gray, (blur_matrix, blur_matrix), 0)

    # Parameters for the Hough circles method
    max_ball_radius = 100
    min_ball_radius = 10
    min_ball_radius = min_r
    max_ball_radius = max_r
    param_1 = 50
    param_2 = 30
    min_dist = gray_blurred.shape[0] // 20  # Height/8
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, minRadius=min_ball_radius, maxRadius=max_ball_radius, param1=param_1, param2=param_2, minDist=min_dist)

    # Create a list to store the circle information
    circle_info = []

    # Draw detected circles and store their coordinates and radii
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for i in detected_circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(img, center, radius, (0, 255, 0), 2)
            cv2.circle(img, center, 1, (0, 0, 255), 3)
            circle_info.append((center, radius))
            if(show): print("                  circle detected")

    if(show): cv2.imshow("Detected Circles", img)
    
    return circle_info

def find_center_handler(image_no):
    path = f"balls/Ball{image_no}.jpg"
    ball_img = cv2.imread(path)
    cv2.imshow("Oringinal Image", ball_img)
    print("Image No: ", image_no)
    find_ball_center(ball_img)

def iterate_through_ball_images():
    for i in range(1, 10):
        print(" ")
        find_center_handler(i)

# iterate_through_ball_images()
# find_center_handler(5)
# image = cv2.imread("balls/Ball1.jpg")
# circle = get_ball_center_coords(image,10,100)
# print(circle)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
