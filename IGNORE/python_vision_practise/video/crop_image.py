import cv2

def crop_image(image, x, y, width, height):
    """
    Crop the input image to the specified region.

    Args:
        image (numpy.ndarray): The input image.
        x (int): X-coordinate of the top-left corner of the region.
        y (int): Y-coordinate of the top-left corner of the region.
        width (int): Width of the region to be cropped.
        height (int): Height of the region to be cropped.

    Returns:
        numpy.ndarray: The cropped image.
    """
    if x < 0:
        x = 0
    if y < 0:
        y = 0

    cropped_image = image[y:y + height, x:x + width]
    return cropped_image

# Example usage:
# Load an image
image_path = "../balls/Ball1.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)

# Define coordinates and dimensions for cropping
x = 100  # X-coordinate of the top-left corner
y = 50   # Y-coordinate of the top-left corner
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Crop the image
cropped = crop_image(image, x, y, width, height)

# Display the cropped image
cv2.imshow("Cropped Image", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
