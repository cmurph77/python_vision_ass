# Performing a perspective transform on the iamge
import cv2
import numpy as np

# Table 3 : [(1044, 581), (3800, 1121), (2987, 2099), (517, 2589)]

img = cv2.imread("tables/Table3.jpg")

tr_x = 1347;
tr_y = 766;

tl_x = 2640;
tl_y = 828;

br_x = 350;
br_y = 3000;

bl_x = 3958;
bl_y = 2963;

pts1 = np.float32([     [208, 1622]  ,  [1459, 685]  ,  [2653, 2806] , [3778, 520]    ])

pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (500, 600))

cv2.imshow("Image", img)
cv2.imshow("Perspective transformation", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
