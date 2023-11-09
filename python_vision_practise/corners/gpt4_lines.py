import cv2
import numpy as np
import matplotlib.pyplot as plt


def perform_blue_mask(img):
    # Define the blue color range in HSV format
    blue_lower = np.array([98, 30, 30])
    blue_upper = np.array([132, 255, 255])

    # Convert the input image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create a mask for the blue color
    mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Perform dilations and erosions on the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Apply the mask to the original image
    blue_masked_image = cv2.bitwise_and(img, img, mask=mask)

    return blue_masked_image


def create_white_mask(image):

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to create a binary image
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    binary_image = cv2.erode(binary_image, None, iterations=3)
    binary_image = cv2.dilate(binary_image, None, iterations=4)

    cv2.imshow("binnary image", binary_image)

    # Invert the binary image (optional, if you want white elements to be white)
    inverted_binary_image = cv2.bitwise_not(binary_image)

    white_masked_image = cv2.bitwise_and(image, image, mask=inverted_binary_image)


    return white_masked_image  # Return the white elements mask


# Load the image
image_path = '../../tables/Table2.jpg'
image = cv2.imread(image_path)

black_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) 


# Convert the image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use the Canny edge detector to find edges in the image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use Hough transform to find lines in the edge-detected image
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Function to draw lines on the image
def draw_lines(black_image, lines, color=[255, 0, 0], thickness=5):
    line_img = np.zeros_like(black_image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    combined = cv2.addWeighted(black_image, 0.8, line_img, 1, 0)
    return combined

# Draw the lines on the original image
lined_image = draw_lines(black_image, lines)

blue_mask = perform_blue_mask(image)
cv2.imshow("blue_mask",blue_mask)
dst = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) 
cv2.bitwise_and(lined_image,blue_mask,dst=dst)
cv2.imshow("blue and lines", dst)

# white_mask = create_white_mask(image)
# cv2.imshow("white_mask",white_mask)
# dst = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) 
# cv2.bitwise_and(lined_image,white_mask,dst=dst)
# cv2.imshow("white and lines",dst)



cv2.imshow("lines_image", lined_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

# Display the resulting image with lines
# plt.imshow(cv2.cvtColor(lined_image, cv2.COLOR_BGR2RGB))
# plt.title('Table Tennis Table with Detected Lines')
# plt.show()
