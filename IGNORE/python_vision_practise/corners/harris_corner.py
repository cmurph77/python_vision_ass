import numpy as np
import cv2

image = cv2.imread('../Table4.jpg') 

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

image_flt = np.float32(gray) 

dst = cv2.cornerHarris(image_flt, 20, 3,10)

dst = cv2.dilate(dst, None) 

image[dst > 0.01 * dst.max()] = [0, 0, 255] 

cv2.imshow('Detected corners', image) 

cv2.waitKey(0)  

cv2.destroyAllWindows() 