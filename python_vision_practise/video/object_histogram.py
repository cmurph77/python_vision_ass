import cv2
import numpy as np
from matplotlib import pyplot as plt
def get_hist(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hue, saturation, value = cv2.split(orange_ball_hsv)
    # Histogram 
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist

ball_img = cv2.imread("/Users/cianmurphy/code_directories/computer_vision/python_vision_ass/balls/orange_ball.jpg")
ball_hist = get_hist(ball_img)



plt.imshow(hist)
cv2.waitKey(0)
cv2.destroyAllWindows()