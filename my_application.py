import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
"""
THIS SCRIPT RUNS ALL 3 PARTS AND PRINTS TO THE CONSOLE.

The balls, tables and data directory must be in the same directory as this file. The same is the case for the TableTennis.avi video file
"""
def main():
    print("STARTING PROGRAM")
    
    print("\nPART 1: ")
    #  Test part 1
    for i in range (1,11) :
        part1(i,False)

    print("\n\nPART 2:")
    for i in range (1,6):
        part2(i,False)

    print("\n\nPART 3:")
    part3(True)
    
# ----------------- PART 1 -------------------------------------------------------------------

# used to detect if the hsv_color is in the range lower - upper
def is_color(hsv_color, lower, upper):
    mask = cv2.inRange(hsv_color, lower, upper)
    return cv2.countNonZero(mask) > 0

# detects the if the color is withint the HSV range for orange
def is_color_orange(hsv_color):
    orange_lower = np.array([10, 100, 100])
    orange_upper = np.array([35, 255, 255])
    return is_color(hsv_color, orange_lower, orange_upper)


# detects the if the color is withint the HSV range for white
def is_color_white(hsv_color):
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 100, 255])
    return is_color(hsv_color, white_lower, white_upper)


# returns the color of the circle in the img
def detect_color(img, x, y, r):
    center = (int(x), int(y)) # create a center object

    # Create a mask with the same dimensions as the input image but only one channel.
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Draw the circle on the mask with white color and filled (-1).
    cv2.circle(mask, center, int(r), (255), -1)

    # Now calculate the mean color using the mask. The mask must be 8-bit single-channel.
    mean_color = cv2.mean(img, mask=mask)

    mean_color_bgr = np.uint8([[mean_color[:3]]])
    mean_color_hsv = cv2.cvtColor(mean_color_bgr, cv2.COLOR_BGR2HSV)

    # print("mean color hsv:",mean_color_hsv, "for circle at x: ",x, ", y: ", y, ", r: ",r)
    return mean_color_hsv

#  set show to True if you want to see the images.
def part1(ball_num,show):
    path = f"balls/Ball{ball_num}.jpg"
    img = cv2.imread(path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_matrix = 5
    gray_blurred = cv2.GaussianBlur(gray, (blur_matrix, blur_matrix), 0)

    # Parameters for the Hough circles method
    max_ball_radius = 200
    min_ball_radius = 10
    param_1 = 90        
    param_2 = 40
    min_dist = gray_blurred.shape[0] // 10  # Height/ 10

    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, minRadius=min_ball_radius,
                                        maxRadius=max_ball_radius, param1=param_1, param2=param_2, minDist=min_dist)

    # Create a list to store the circle information
    circle_info = []

    # Draw detected circles and store their coordinates and radii
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for i in detected_circles[0, :]:
            x = i[0]
            y = i[1]
            center = (x, y)
            radius = i[2]
            color = detect_color(img, x, y, radius)
            if is_color_orange(color) or is_color_white(color):
                cv2.circle(img, center, radius, (255, 0, 0), 2)
                cv2.circle(img, center, 1, (0, 0, 255), 3)
                circle_info.append((center, radius))

    print("Image: ", ball_num, " => ",circle_info)

    if show : cv2.imshow("Detected Circles", img)

    if show: cv2.waitKey(0)
    if show: cv2.destroyAllWindows()

# ----------------- PART 2 -------------------------------------------------------------------

# Function to extend a line segment to the image border
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

def part2(table_num, show):
    min_distance = 300
    blue_lower = np.array([95, 20, 20])
    blue_upper = np.array([135, 255, 255])
    min_line_length = 130

    # Read in the image
    image_path = f'tables/Table{table_num}.jpg'
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
    if show : cv2.imshow("BLUE MASK pre processed 4.1", mask)

    # Perform dilations and erosions on the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    blue_mask = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow("BLUE MASK & ORIGINAL", blue_mask)
    if show : cv2.imshow("BLUE MASK processed 4.2", mask)

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

    if show : cv2.imshow("CONTOURS IMAGE 4.3", contour_img)

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


    if show : cv2.imshow("PRE PROCCESSED INTERSECTING LINES IMAGE 4.4",
            intersecting_lines_image)

    # ----------------------------------------------------------
    # PROCESS THE INTERSECTUNG LINES IMAGE
    #   this turns the multipls lines that may be extended out into one solid line

    intersecting_lines_image = cv2.dilate(
        intersecting_lines_image, None, iterations=10)
    intersecting_lines_image = cv2.erode(
        intersecting_lines_image, None, iterations=14)

    if show : cv2.imshow("POST PROCCESSED INTERSECTING LINES IMAGE 4.5",
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
    print("Corners for Table:", table_num, " => ",filtered_nearby_coordinates)

    # # ----------------------------------------------------------
    # DRAW CORNERS

    # # Draw the filtered corners on the intersecting lines image
    # for x, y in filtered_nearby_coordinates:
    #     cv2.circle(nearby_filtered_corners, (x, y), 15, (0, 0, 255), -1)

    # Draw corners on the original image
    for x, y in filtered_nearby_coordinates:
        cv2.circle(img, (x, y), 15, (0, 0, 255), -1)

    # cv2.imshow("nearby filtered coords", nearby_filtered_corners)

    if show : cv2.imshow("oringal image with corners marked 4.6", img)

    #  ----------------------------------------------------------
    # PERFROM PERSPECTTIVE TRANSFORM

    if len(filtered_nearby_coordinates) == 4 :
        pts1 = np.float32(filtered_nearby_coordinates)

        pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(img, matrix, (500, 600))

        if show : cv2.imshow("PERSPECTIVE TRANSFORMATION 4.7", result)
    else: print("     Not enought points for transform in image ", table_num)
    # ----------------------------------------------------------
    if show: cv2.waitKey(0)
    if show: cv2.destroyAllWindows()

# ----------------- PART 3 -------------------------------------------------------------------

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


""""This function tracks the trajectory of the ball and detects whether is bounces or gets hit by player"""
def track_trajectory(locations,video_width,print_result):
    print("\nRESULTS:")
    # Initialize variables to keep track of the ball's direction
    vertical_direction = None  # Start with no vertical direction
    horizontal_direction = None  # Start with no horizontal direction
    direction_changes = []
    print(direction_changes)

    # We will compare each point with the previous one to determine direction changes
    # Start with the second point (index 1) because we need a previous point for comparison
    for i in range(1, len(locations)):
        # Current and previous points
        current_frame, current_point = locations[i]
        x_loc = current_point[0]
        y_loc = current_point[1]

        prev_frame, prev_point = locations[i - 1]

        # Determine the vertical and horizontal directions of movement
        current_vertical_direction = "down" if current_point[1] > prev_point[1] else "up"
        current_horizontal_direction = "right" if current_point[0] > prev_point[0] else "left"

        # Check if there has been a change in vertical direction to "up"
        if current_vertical_direction == "up" and vertical_direction == "down":
            out = f"Frame:  {current_frame} ({y_loc} , {x_loc}) Bounce on the Table" 
            if print_result: print(out)           

        # Check if there has been a change in horizontal direction
        if current_horizontal_direction != horizontal_direction and horizontal_direction is not None:
            direction_changes.append((current_frame, "HIT BY PLAYER"))
            center_margin = 50
            if x_loc > video_width/2 - center_margin and x_loc < video_width/2 + center_margin  :  # changes direction in center - must bnet
                out = f"Frame: {current_frame} ({y_loc} , {x_loc}) Hit the Net" 
                if print_result: print(out)
            else:  
                out = f"Frame: {current_frame} ({y_loc} , {x_loc}) Hit By Player" 
                if print_result: print(out)
        
        # Update the current directions
        vertical_direction = current_vertical_direction
        horizontal_direction = current_horizontal_direction


def part3(show):

    print("\nANALYSING VIDEO. PLEASE WAIT...")
    if show: print("PRESS 'ESC' BUTTON TO HALT VIDEO ANALYSIS AND SHOW EVENT RESULTS")
    # Create a VideoCapture object to read from a video file or camera
    video_source = 'TableTennis.avi'  # Replace with your video source
    cap = cv2.VideoCapture(video_source)

    # Get histograms for the players cloths and skin
    player_1 = cv2.imread("data/p1_hist.jpg")
    p1_hist = get_hist(player_1)
    player_2 = cv2.imread("data/p2_hist.jpg")
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
        # print("frame no:", frame_number)
        ret, frame = cap.read()                 # get the next frame
        if not ret: break
    

        # Apply the background subtractor to get the foreground mask
        fg_mask = bg_subtractor.apply(frame)



        # # Post-process the mask with erosion and dilations
        # fg_mask = cv2.erode(fg_mask, None, iterations=3)
        # fg_mask = cv2.dilate(fg_mask, None, iterations=40)
        # fg_mask = cv2.erode(fg_mask, None, iterations=20)
        # fg_mask = cv2.dilate(fg_mask, None, iterations=1)
        

        # Post-process the mask with erosion and dilations
        fg_mask = cv2.erode(fg_mask, None, iterations=3)
        fg_mask = cv2.dilate(fg_mask, None, iterations=40)
        fg_mask = cv2.erode(fg_mask, None, iterations=20)
        fg_mask = cv2.dilate(fg_mask, None, iterations=1)
        if show : cv2.imshow("post processed mask",fg_mask)

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
                        # print("cX: ", cX, "cY: ", cY)
                    else: 
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # player detected so draw red box around contour


        # Display the original frame and the result
        if show: cv2.imshow('Original Video', frame)
        

        if show : 
            if cv2.waitKey(30) & 0xFF == 27: break  # Press 'ESC' button to exit the while loop


    cap.release()
    cv2.destroyAllWindows()

    filled_gaps_locations = fill_gaps(ball_locations) # locations of the ball for each frame
    track_trajectory(filled_gaps_locations,video_width,True)   # analyse ball path and detect bounces

    if show:
        cv2.imshow("COMPLETE PATH OF BALL", draw_path(filled_gaps_locations))
        print("\nPRESS 'ESC' TO QUIT")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
# ----------------- MAIN -------------------------------------------------------------------

if __name__ == "__main__":
    main()