import cv2
import numpy as np
import math


# Load the image
image_path = '../../tables/Table4.jpg'
image = cv2.imread(image_path)
black_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) 



#----------------------- Detect straight Lines in an image ---------------------------------------------------------------------------
lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) 

# find the edges in the image of the largest contours
edges = cv2.Canny(image, 150, 250, apertureSize=3)

# Apply the Hough Line Transform on the detected edges
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    
# filter out the small lines
min_line_length = 200
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        x1,y1 = l[0], l[1]
        x2,y2 = l[2], l[3]
        # Calculate the Euclidean distance
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        # Round the distance to the nearest whole number
        line_length = round(distance)
        if line_length > min_line_length:
            cv2.line(lines_image, (x1,y1), (x2,y2), (0,0,255), 3, cv2.LINE_AA) # draw a line between x1,y1 and x2,y2

cv2.imshow("lines_image", lines_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

