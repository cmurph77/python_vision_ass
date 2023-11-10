import cv2
import numpy as np

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

def is_player(contour_hist,p1_hist,p2_hist):
    distance_p1 = cv2.compareHist(contour_hist, p1_hist, cv2.HISTCMP_BHATTACHARYYA)
    distance_p2 = cv2.compareHist(contour_hist, p2_hist, cv2.HISTCMP_BHATTACHARYYA) 

    min_distance = float(0.94)
    if distance_p1 < min_distance or distance_p2 < min_distance:
        print("distance p1",distance_p1)
        print("distance p2",distance_p1)

        return True
    else:
        return False

def is_ball(contour_hist):
    orange_ball_img = cv2.imread("balls/orange_ball.jpg")
    ball_hist = get_hist(orange_ball_img)

    min_distance = float(0.98)


    distance = cv2.compareHist(contour_hist, ball_hist, cv2.HISTCMP_BHATTACHARYYA)
    print("distance ball", distance)
    if distance > min_distance:
        return True
    else: 
        return False


# Create a VideoCapture object to read from a video file or camera
video_source = 'TableTennis.avi'  # Replace with your video source
cap = cv2.VideoCapture(video_source)

player_1 = cv2.imread("tables/p1_hist.jpg")
p1_hist = get_hist(player_1)

player_2 = cv2.imread("tables/p2_hist.jpg")
p2_hist = get_hist(player_2)

ball_locations = []

# Initialize the background subtractor with GMM
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=10, varThreshold=30 )

frame_number = 0

video_height = 540
video_width = 960
ball_path = np.zeros((video_height, video_width, 3), dtype=np.uint8)

while True:
    frame_number = frame_number + 1
    print("frame: ", frame_number)
    ret, frame = cap.read()
    if not ret:
        break

    tracking_img = frame.copy()

    # Apply the background subtractor to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)


    # Post-process the mask (optional)
    fg_mask = cv2.erode(fg_mask, None, iterations=3)
    fg_mask = cv2.dilate(fg_mask, None, iterations=40)
    fg_mask = cv2.erode(fg_mask, None, iterations=20)
    fg_mask = cv2.dilate(fg_mask, None, iterations=1)

    # cv2.imshow("fg_mask",fg_mask)

    # Find contours in the mask to detect moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_histograms = []
    index = 0


    # Draw bounding boxes around detected objects
    for contour in contours:
        # Calculate the moments
        M = cv2.moments(contour)
        # Calculate the center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        margin = 150
        if cX > margin and cX < video_width - margin: 
            if cv2.contourArea(contour) > 2500 and cv2.contourArea(contour) < 10000:  # Adjust the area threshold as needed
                x, y, w, h = cv2.boundingRect(contour)
                contour_image = crop_image(frame,x,y,w,h,)
                # cv2.imshow("contour image",contour_image)
                contour_hist = get_hist(contour_image)
                # if is_ball(contour_hist):
                player = is_player(contour_hist,p1_hist,p2_hist)
                if not player :
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # draw green box around ball
                
                    cv2.circle(tracking_img, (cX, cY), 5, (255, 0, 0), -1)
                    cv2.circle(ball_path, (cX, cY), 5, (255, 0, 0), -1)
                    ball_locations.append((frame_number,(cX,cY)))
                else: 
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # Display the original frame and the result
    cv2.imshow('Original Video', frame)
    # cv2.imshow('Tracking Frame', tracking_img)
    cv2.imshow('Ball Path', ball_path)


    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        print(ball_locations)
        break

cap.release()
cv2.destroyAllWindows()