import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

# Function to extend a line segment to the image border TODO review this function
""""
Table 5: [(1208, 478), (2518, 542), (3714, 2637), (278, 2685)]
Table 2: [(3778, 520), (1459, 685), (208, 1622), (2653, 2806)]
"""
min_distance = 300
blue_lower = np.array([95, 20, 20])
blue_upper = np.array([135, 255, 255])
min_line_length = 130

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


# Read in the image
image_path = 'tables/Table4.jpg'
img = cv2.imread(image_path)

# ----------------------------------------------------------
# CREATE A MASK FOR BLUE REGIONS (TABLE TOP COLOR)

# # Define the blue color range in HSV format
# blue_lower = np.array([95, 20, 20])
# blue_upper = np.array([135, 255, 255])

# Convert the input image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create a mask for the blue color
mask = cv2.inRange(hsv, blue_lower, blue_upper)
cv2.imshow("BLUE MASK pre processed 4.1", mask)

# Perform dilations and erosions on the mask
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)


blue_mask = cv2.bitwise_and(img, img, mask=mask)

# cv2.imshow("BLUE MASK & ORIGINAL", blue_mask)
cv2.imshow("BLUE MASK processed 4.2", mask)

# ----------------------------------------------------------
# FIND CONTOURS IN THE MASK

# Process the mask to join contours together
mask = cv2.dilate(mask, None, iterations=20)
mask = cv2.erode(mask, None, iterations=20)
mask = cv2.dilate(mask, None, iterations=10)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image to draw contours on
contour_img = np.zeros_like(mask)

# Draw the contours on the image
cv2.drawContours(contour_img, contours, -1, (255), 2)

cv2.imshow("CONTOURS IMAGE 4.3", contour_img)

# ----------------------------------------------------------
# DETECT LINES IN THE CONTOUR IMAGE AND EXTEND LINES TO THE EDGE OF THE IMAGE

# Recreate an image to draw lines on, using the size of the contour image
intersecting_lines_image = np.zeros(
    (img.shape[0], img.shape[1], 3), dtype=np.uint8)

# Find the edges in the image of the largest contour
edges = cv2.Canny(contour_img, 150, 250, apertureSize=3)

# Apply the Hough Line Transform on the detected edges
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

# Filter out the small lines
# min_line_length = 150
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        x1, y1, x2, y2 = l[0], l[1], l[2], l[3]
        # Calculate the Euclidean distance to filter out small lines
        if math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) > min_line_length:
            # Extend the line to the image borders
            x1_ext, y1_ext, x2_ext, y2_ext = extend_line(
                x1, y1, x2, y2, contour_img.shape[1], contour_img.shape[0])
            # Draw the extended line
            cv2.line(intersecting_lines_image, (x1_ext, y1_ext),
                     (x2_ext, y2_ext), (255, 255, 255), 10, cv2.LINE_AA)


cv2.imshow("PRE PROCCESSED INTERSECTING LINES IMAGE 4.4",
           intersecting_lines_image)

# ----------------------------------------------------------
# PROCESS THE INTERSECTUNG LINES IMAGE
#   this turns the multipls lines that may be extended out into one solid line

intersecting_lines_image = cv2.dilate(
    intersecting_lines_image, None, iterations=10)
intersecting_lines_image = cv2.erode(
    intersecting_lines_image, None, iterations=14)

cv2.imshow("POST PROCCESSED INTERSECTING LINES IMAGE 4.5",
           intersecting_lines_image)

# make copies of the itersecting lines for later to draw corners
border_filtered_corners = intersecting_lines_image.copy()
nearby_filtered_corners = intersecting_lines_image.copy()

# ----------------------------------------------------------
# DETECT CORNERS IN THE INTERSECTING LINES

# convert image to gray to process
gray = cv2.cvtColor(intersecting_lines_image, cv2.COLOR_BGR2GRAY)

# Detect corners
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


# cv2.imshow("CORNERS DETECTED 4.6", intersecting_lines_image)

# ----------------------------------------------------------
# FILTER OUT THE COORDS THAT MAY BE ON THE EDGE OF THE IMAGE (THESE ARE ERRORS)

filtered_border_coordinates = []

# Define the margin distance from the image edge
margin = 100

# Loop through the list of corner coordinates
for x, y in corner_coordinates:
    # Check if the corner is more than 'margin' pixels away from any of the edges
    if x > margin and y > margin and x < intersecting_lines_image.shape[1] - margin and y < intersecting_lines_image.shape[0] - margin:
        filtered_border_coordinates.append((x, y))


# Draw the filtered corners on the image
for x, y in filtered_border_coordinates:
    cv2.circle(border_filtered_corners, (x, y), 5, (255, 0, 0), -1)

# cv2.imshow("border filtered coords", border_filtered_corners)

# ----------------------------------------------------------
# FILTER OUT THE COORDS THAT ARE NEARBY TO EACHOTHER (there will be four corners on each line intersection)

filtered_nearby_coordinates = []

# Define the minimum distance between points

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

# Return the list of filtered coordinates for further use
print(filtered_nearby_coordinates)

# # ----------------------------------------------------------
# DRAW CORNERS

# # Draw the filtered corners on the intersecting lines image
# for x, y in filtered_nearby_coordinates:
#     cv2.circle(nearby_filtered_corners, (x, y), 15, (0, 0, 255), -1)

# Draw corners on the original image
for x, y in filtered_nearby_coordinates:
    cv2.circle(img, (x, y), 15, (0, 0, 255), -1)

# cv2.imshow("nearby filtered coords", nearby_filtered_corners)

cv2.imshow("oringal image with corners marked 4.6", img)

#  ----------------------------------------------------------
# PERFROM PERSPECTTIVE TRANSFORM

# # remove before submitting
# pts1 = [(1044, 581), (517, 2589), (3800, 1121), (2987, 2099)]

# pts1 = np.float32(filtered_nearby_coordinates)

# pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# result = cv2.warpPerspective(img, matrix, (500, 600))

# cv2.imshow("PERSPECTIVE TRANSFORMATION 4.7", result)
# ----------------------------------------------------------
cv2.waitKey(0)
cv2.destroyAllWindows()
