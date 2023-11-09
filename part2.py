import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Function to extend a line segment to the image borders
def extend_line(x1, y1, x2, y2, width, height):
    if x2 != x1:
        # Calculate the slope and y-intercept of the line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Get the extended points on the image border
        y_start = int(slope * 0 + intercept)
        y_end = int(slope * width + intercept)
        
        # Handle cases where the line might go beyond the image height
        if y_start > height:
            y_start = height
            x_start = int((y_start - intercept) / slope)
        elif y_start < 0:
            y_start = 0
            x_start = int((y_start - intercept) / slope)
        else:
            x_start = 0

        if y_end > height:
            y_end = height
            x_end = int((y_end - intercept) / slope)
        elif y_end < 0:
            y_end = 0
            x_end = int((y_end - intercept) / slope)
        else:
            x_end = width
    else:
        # The line is vertical, so we make it span the height of the image
        x_start, x_end = x1, x1
        y_start, y_end = 0, height

    return x_start, y_start, x_end, y_end

# table_5_points = [(1224, 490), (2562, 555), (223, 2739), (3742, 2762)] # 1224	490	2562	555	223	2739	3742	2762

# Read in the image
image_path = 'tables/Table3.jpg'
img = cv2.imread(image_path)

# ----------------------------------------------------------
# CREATE A MASK FOR BLUE REGIONS (TABLE TOP COLOR) 

# Define the blue color range in HSV format
blue_lower = np.array([95, 20, 20])
blue_upper = np.array([135, 255, 255])

# Convert the input image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create a mask for the blue color
mask = cv2.inRange(hsv, blue_lower, blue_upper)

# Perform dilations and erosions on the mask
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

# Apply the mask to the original image
# blue_masked_image = cv2.bitwise_and(img, img, mask=mask)


# ----------------------------------------------------------
# FIND CONTOURS IN THE MASK

# Process the mask to join contours together
mask = cv2.dilate(mask,None, iterations=10)
mask = cv2.erode(mask,None, iterations=10)
mask = cv2.dilate(mask,None, iterations=2)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image to draw contours on
contour_img = np.zeros_like(mask)

# Draw the contours on the image
cv2.drawContours(contour_img, contours, -1, (255), 2)

# cv2.imshow("contour image", contour_img)

# ----------------------------------------------------------
#  Largest contour

# Find the largest contour
cnt = max(contours, key=cv2.contourArea)

# Create an empty image to draw contours on
largest_contour_img = np.zeros_like(mask)

# Draw the contours on the image
cv2.drawContours(largest_contour_img, cnt, -1, (255), 2)

cv2.imshow("largest_contour",largest_contour_img)

largest_contour_img = cv2.dilate(largest_contour_img,None, iterations=5)
largest_contour_img = cv2.erode(largest_contour_img,None, iterations=5)

cv2.imshow("post processed  largest_contour",largest_contour_img)

# ----------------------------------------------------------
# Detect lines and extend them to edges

# Recreate an image to draw lines on, using the size of the contour image
intersecting_lines_image = np.zeros((contour_img.shape[0], contour_img.shape[1], 3), dtype=np.uint8)

# Find the edges in the image of the largest contour
edges = cv2.Canny(contour_img, 150, 250, apertureSize=3)

# Apply the Hough Line Transform on the detected edges
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

# Filter out the small lines
min_line_length = 200
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
        # Calculate the Euclidean distance to filter out small lines
        if math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) > min_line_length:
            # Extend the line to the image borders
            x1_ext, y1_ext, x2_ext, y2_ext = extend_line(x1, y1, x2, y2, contour_img.shape[1], contour_img.shape[0])
            # Draw the extended line
            cv2.line(intersecting_lines_image, (x1_ext, y1_ext), (x2_ext, y2_ext), (255, 255, 255), 10, cv2.LINE_AA)
            

cv2.imshow("interecting lines", intersecting_lines_image)

# ----------------------------------------------------------
# process the exteneded lines image

intersecting_lines_image = cv2.dilate(intersecting_lines_image,None, iterations=10)
intersecting_lines_image = cv2.erode(intersecting_lines_image,None, iterations=14)

cv2.imshow("Post processed intersecting lines image", intersecting_lines_image)
border_filtered_corners = intersecting_lines_image.copy()
nearby_filtered_corners = intersecting_lines_image.copy()

# ----------------------------------------------------------
# detect corners in the intersecting lines image

gray = cv2.cvtColor(intersecting_lines_image, cv2.COLOR_BGR2GRAY) 

# Continue from the previous Harris corner detection process
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst_dilated = cv2.dilate(dst, None)

# Set threshold to detect corners
threshold = 0.03 * dst.max()

# Initialize list to store coordinates of the corners
corner_coordinates = []

# Loop through all points in the dilated destination image
for i in range(dst_dilated.shape[0]):
    for j in range(dst_dilated.shape[1]):
        if dst_dilated[i, j] > threshold:
            # This point is considered a corner
            corner_coordinates.append((j, i))  # Append as (x, y)
            cv2.circle(intersecting_lines_image, (j, i), 5, (255, 0, 0), -1)

# # Print out the coordinates of the corners
# for coord in corner_coordinates:
#     print("Corner detected at: x = {}, y = {}".format(coord[0], coord[1]))

cv2.imshow("Corners detected", intersecting_lines_image)

            
# ----------------------------------------------------------
# filter out the coords at the edge of the image
# Initialize list to store filtered corners
filtered_border_coordinates = []

# Define the margin distance from the image edge
margin = 100

# Loop through the list of corner coordinates
for x, y in corner_coordinates:
    # Check if the corner is more than 'margin' pixels away from any of the edges
    if x > margin and y > margin and x < intersecting_lines_image.shape[1] - margin and y < intersecting_lines_image.shape[0] - margin:
        filtered_border_coordinates.append((x, y))

# Now 'filtered_corner_coordinates' contains corners that are not near the edges of the image
# # Let's print them out
# for coord in filtered_border_coordinates:
#     # print("Filtered corner: x = {}, y = {}".format(coord[0], coord[1]))

# Draw the filtered corners on the image
for x, y in filtered_border_coordinates:
    cv2.circle(border_filtered_corners, (x, y), 5, (255, 0, 0), -1)

cv2.imshow("border filtered coords", border_filtered_corners)

# ----------------------------------------------------------
# Filter out any points that are close together
# Given a list of coordinates, filter out those that are within 100 pixels of each other

# Initialize list to store filtered coordinates
filtered_nearby_coordinates = []

# Define the minimum distance between points
min_distance = 200

# Loop through the list of coordinates
for coord in filtered_border_coordinates:
    x, y = coord
    # Assume the point is far enough away until we check it
    far_enough = True
    # Check the distance of the current coordinate from all coordinates in filtered_coordinates
    for filtered_coord in filtered_nearby_coordinates:
        fx, fy = filtered_coord
        # Calculate Euclidean distance
        distance = np.sqrt((fx - x) ** 2 + (fy - y) ** 2)
        # If a point is found within min_distance, set far_enough to False and break
        if distance < min_distance:
            far_enough = False
            break
    # If the point is far enough from all others, add to the list of filtered coordinates
    if far_enough:
        filtered_nearby_coordinates.append(coord)

print(filtered_nearby_coordinates)  # Return the list of filtered coordinates for further use

# # ----------------------------------------------------------
# draw corners

# Draw the filtered corners on the intersecting lines image
for x, y in filtered_nearby_coordinates:
    cv2.circle(nearby_filtered_corners, (x, y), 15, (0, 0, 255), -1)

for x, y in filtered_nearby_coordinates:
    cv2.circle(img, (x, y), 15, (0, 0, 255), -1)

# cv2.imshow("nearby filtered coords", nearby_filtered_corners)

cv2.imshow("oringal image with corners marked",img)

#  ----------------------------------------------------------
# Perform perspective transform
# pts1 = np.float32(filtered_nearby_coordinates)
# pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# result = cv2.warpPerspective(img, matrix, (500, 600))

# cv2.imshow("Image", img)
# cv2.imshow("Perspective transformation", result)

#  ----------------------------------------------------------

cv2.waitKey(0)
cv2.destroyAllWindows()




