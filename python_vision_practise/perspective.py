# Performing a perspective transform on the iamge
import cv2
import numpy as np

img = cv2.imread("Table4.jpg")

tr_x = 1347;
tr_y = 766;

tl_x = 2640;
tl_y = 828;

br_x = 350;
br_y = 3000;

bl_x = 3958;
bl_y = 2963;

pts1 = np.float32([[tr_x, tr_y], [tl_x, tl_y], [br_x, br_y], [bl_x, bl_y]])

pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (500, 600))

cv2.imshow("Image", img)
cv2.imshow("Perspective transformation", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
