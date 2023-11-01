import numpy as np 
import cv2

img = cv2.imread('Table4.jpg', cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1)

canny = cv2.Canny(img,1000,200)

cv2.imshow("sobel x",sobelx)
cv2.imshow("sobel y", sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()
