import cv2

def get_hist(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hue, saturation, value = cv2.split(orange_ball_hsv)
    # Histogram 
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist

def crop_image(image, x, y, width, height):

    if x < 0:
        x = 0
    if y < 0:
        y = 0

    cropped_image = image[y:y + height, x:x + width]
    return cropped_image

def find_closest_histogram(image_hist, histogram_array):
    closest_index = -1
    min_distance = float('inf')

    for i, hist in enumerate(histogram_array):
        # Calculate the Bhattacharyya distance between the image's histogram and each histogram in the array
        distance = cv2.compareHist(image_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

        # Check if the current histogram is closer than the previous closest one
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    # return closest_index, min_distance
    return closest_index

orange_ball_img = cv2.imread("../balls/orange_ball.jpg")
ball_hist = get_hist(orange_ball_img)

# Create a VideoCapture object to read from a video file or camera
video_source = 'TableTennis.avi'  # Replace with your video source
cap = cv2.VideoCapture(video_source)

# Initialize the background subtractor with GMM
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=20, varThreshold=30 )

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the background subtractor to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Post-process the mask (optional)
    fg_mask = cv2.erode(fg_mask, None, iterations=3)
    fg_mask = cv2.dilate(fg_mask, None, iterations=3)

    # Find contours in the mask to detect moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_histograms = []
    index = 0

    # Draw bounding boxes around detected objects
    for contour in contours:
        if cv2.contourArea(contour) > 80:  # Adjust the area threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            contour_image = crop_image(frame,x,y,w,h,)
            cv2.imshow("contour image",contour_image)
            contour_hist = get_hist(contour_image)
            frame_histograms.append(contour_hist)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            index = index +1

    # find the closest histogram
    hist_index = find_closest_histogram(ball_hist,frame_histograms)
    # print("Frame No:", frame_number, hist_index)

    # Display the original frame and the result
    cv2.imshow('Original Video', frame)


    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
