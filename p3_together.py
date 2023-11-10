import cv2
import numpy as np

show = True
"""This method draws the ball path coordinates onto an empty black image"""
def draw_path(a):    
    video_height = 540
    video_width = 960
    path_img = np.zeros((video_height, video_width, 3), dtype=np.uint8) #create an empty black img to draw path on
    first = True
    for  point in a:
        current_center = point[1]
        if first: first = False   # no longer on the first frame so set to false
        else: cv2.line(path_img, prev_center,current_center,(255,0,0),2)   # we only want to draw lines from 2nd frame onwards
        cv2.circle(path_img, current_center, 3, (0,0,255), -1)  # draw circle on the image

        prev_center = current_center
    return path_img


""" A function to perform linear interpolation between two points"""
def interpolate(start_frame, end_frame, start_point, end_point):
    if start_frame == end_frame:
        return [start_point]
    frame_coords = []
    delta = (end_frame - start_frame)
    delta_x = (end_point[0] - start_point[0]) / delta
    delta_y = (end_point[1] - start_point[1]) / delta
    for i in range(1, delta):
        new_coords = (int(start_point[0] + delta_x * i), int(start_point[1] + delta_y * i))
        frame_coords.append((start_frame + i, new_coords))
    return frame_coords


"""this funciton fills missing points between 2 poitns with a straight line"""
def fill_gaps(ball_locations):
    # Fill in the coordinates for each frame
    filled_coordinates = []
    for i in range(len(ball_locations)):
        frame, coords = ball_locations[i]
        # If it's the last point or points have the same frame number, just add the point
        if i == len(ball_locations) - 1 or ball_locations[i][0] == ball_locations[i+1][0]:
            filled_coordinates.append((frame, coords))
        else:
            # Get the next point to calculate the line
            next_frame, next_coords = ball_locations[i+1]
            # Add the current point
            filled_coordinates.append((frame, coords))
            # Interpolate between the current point and the next point
            filled_coordinates.extend(interpolate(frame, next_frame, coords, next_coords))

    return filled_coordinates

"""" This function returns the histogram of an image"""
def get_hist(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hue, saturation, value = cv2.split(orange_ball_hsv)
    # Histogram 
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist

""""This function returns a cropped image to the size specified in parameters"""
def crop_image(image, x, y, width, height):

    if x < 0:
        x = 0
    if y < 0:
        y = 0

    cropped_image = image[y:y + height, x:x + width]
    return cropped_image


""""this funciton compares the contour histogram to the histograms of the players and returns True if contour is player"""
def is_player(contour_hist,p1_hist,p2_hist):
    distance_p1 = cv2.compareHist(contour_hist, p1_hist, cv2.HISTCMP_BHATTACHARYYA)
    distance_p2 = cv2.compareHist(contour_hist, p2_hist, cv2.HISTCMP_BHATTACHARYYA) 

    min_distance = float(0.94)
    if distance_p1 < min_distance or distance_p2 < min_distance:


        return True
    else:
        return False


#  TODO COMMENT TRAJECTORY
""""This function tracks the trajectory of the ball and detects whether is bounces or gets hit by player"""
def track_trajectory(locations):
    # Initialize variables to keep track of the ball's direction
    vertical_direction = None  # Start with no vertical direction
    horizontal_direction = None  # Start with no horizontal direction
    direction_changes = []

    # We will compare each point with the previous one to determine direction changes
    # Start with the second point (index 1) because we need a previous point for comparison
    for i in range(1, len(locations)):
        # Current and previous points
        current_frame, current_point = locations[i]
        prev_frame, prev_point = locations[i - 1]

        # Determine the vertical and horizontal directions of movement
        current_vertical_direction = "down" if current_point[1] > prev_point[1] else "up"
        current_horizontal_direction = "right" if current_point[0] > prev_point[0] else "left"

        # # Check if there has been a change in vertical direction to "up"
        # if current_vertical_direction == "up" and vertical_direction == "down":
        #     direction_changes.append((current_frame, "BOUNCE"))

        # # Check if there has been a change in horizontal direction
        # if current_horizontal_direction != horizontal_direction and horizontal_direction is not None:
        #     direction_changes.append((current_frame, "PADDLE"))
        
        # Update the current directions
        vertical_direction = current_vertical_direction
        horizontal_direction = current_horizontal_direction

    # Output the direction changes
    for change in direction_changes:
        print(f"Frame {change[0]}: {change[1]}")

# Create a VideoCapture object to read from a video file or camera
video_source = 'TableTennis.avi'  # Replace with your video source
cap = cv2.VideoCapture(video_source)

# Get histograms for the players cloths and skin
player_1 = cv2.imread("tables/p1_hist.jpg")
p1_hist = get_hist(player_1)
player_2 = cv2.imread("tables/p2_hist.jpg")
p2_hist = get_hist(player_2)


ball_locations = []                          # initialize list of ball locations over frames

# Initialize the background subtractor with GMM
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=10, varThreshold=30 )  

frame_number = 0                            # itialise the frame counter
video_height,video_width = 540,960          # set the height and width of the video frames
ball_path = np.zeros((video_height, video_width, 3), dtype=np.uint8) # create an empty black image to draw ball path

# This while loops through the whole video
while True:
    frame_number = frame_number + 1         # update frame count
    print("frame no:", frame_number)
    ret, frame = cap.read()                 # get the next frame
    if not ret: break
 

    # Apply the background subtractor to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)


    # Post-process the mask with erosion and dilations
    fg_mask = cv2.erode(fg_mask, None, iterations=3)
    fg_mask = cv2.dilate(fg_mask, None, iterations=40)
    fg_mask = cv2.erode(fg_mask, None, iterations=20)
    fg_mask = cv2.dilate(fg_mask, None, iterations=1)

    # Find contours in the mask to detect moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # frame_histograms = []
    index = 0


    # Draw bounding boxes around detected objects
    for contour in contours:
        
        # Calculate the center of the contour
        M = cv2.moments(contour)                 # Calculate the moments for center calculation 
        if M["m00"] != 0:             
            cX = int(M["m10"] / M["m00"])         # x coord for center
            cY = int(M["m01"] / M["m00"])         # y coord for center
        
        margin = 180                               # this is the margin on left and rigth where players are so can ignore movement here
        if cX > margin and cX < video_width - margin:                             # check if the contour is outside the left and right margin
            if cv2.contourArea(contour) > 2500 and cv2.contourArea(contour) < 10000:    # Adjust the area threshold as needed
                x, y, w, h = cv2.boundingRect(contour)                            # draw bounding rectangle around contour
                contour_image = crop_image(frame,x,y,w,h,)                        # create an image with just the contour
                contour_hist = get_hist(contour_image)                            # get a histogram for the contour
                player = is_player(contour_hist,p1_hist,p2_hist)                  # check if the contour hist is close to either player hist
                if not player :                                                   # not a player so should be ball
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw green box around ball
                    cv2.circle(ball_path, (cX, cY), 5, (255, 0, 0), -1)           # mark the ball path image with contour center
                    ball_locations.append((frame_number,(cX,cY)))                 # note the ball center
                else: 
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # player detected so draw red box around contour


    # Display the original frame and the result
    if show: cv2.imshow('Original Video', frame)
    if show: cv2.imshow('Ball Path', ball_path)

    if cv2.waitKey(30) & 0xFF == 27: break  # Press 'ESC' button to exit the while loop


cap.release()
cv2.destroyAllWindows()

filled_gaps_locations = fill_gaps(ball_locations)
# print(filled_gaps_locations)
track_trajectory(filled_gaps_locations)

cv2.imshow("filled coords", draw_path(filled_gaps_locations))

# print(filled_coords)

cv2.waitKey(0)
cv2.destroyAllWindows()