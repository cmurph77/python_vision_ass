import cv2

def get_hist(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hue, saturation, value = cv2.split(orange_ball_hsv)
    # Histogram 
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist


# Create a VideoCapture object to read from a video file or camera
video_source = 'TableTennis.avi'  # Replace with your video source
cap = cv2.VideoCapture(video_source)

# Initialize the background subtractor with GMM
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Post-process the mask (optional)
    fg_mask = cv2.erode(fg_mask, None, iterations=2)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    # Find contours in the mask to detect moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame and the result
    cv2.imshow('Original Video', frame)
    og_masked_image = cv2.bitwise_and(frame, frame, mask=fg_mask)
    cv2.imshow("og masked image",og_masked_image)

    cv2.imshow('Foreground Mask', fg_mask)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
