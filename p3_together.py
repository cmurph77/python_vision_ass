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

"""ball_locations = [(21, (667, 408)), (22, (639, 386)), (23, (612, 369)), (24, (766, 367)), (25, (769, 368)),
                   (25, (559, 342)), (26, (770, 366)), (26, (534, 335)), (27, (768, 366)), (28, (793, 332)), 
                   (34, (360, 380)), (35, (341, 398)), (36, (322, 392)), (37, (305, 376)), (46, (358, 294)), 
                   (49, (449, 293)), (50, (478, 297)), (51, (507, 303)), (54, (590, 338)), (56, (645, 371)),
                     (68, (725, 308)), (69, (670, 285)), (72, (788, 320)), (73, (798, 324)), (74, (807, 329)), 
                     (76, (492, 337)), (78, (445, 376)), (79, (421, 398)), (80, (401, 401)), (81, (387, 383)), 
                     (92, (294, 325)), (93, (313, 324)), (94, (386, 319)), (95, (412, 321)), (98, (487, 341)), 
                     (100, (536, 368)), (101, (559, 384)), (102, (581, 403)), (103, (599, 404)), (104, (611, 387)),
                       (120, (719, 356)), (124, (379, 315)), (125, (351, 331)), (126, (324, 350)), (127, (296, 372)),
                         (127, (802, 365)), (128, (269, 396)), (129, (241, 424)), (130, (214, 458)), (131, (197, 437)),
                           (132, (182, 415)), (133, (168, 398)), (134, (152, 382)), (143, (200, 315)), (144, (242, 305)),
                             (145, (282, 299)), (146, (321, 297)), (147, (358, 297)), (148, (393, 301)), (149, (426, 307)),
                               (150, (457, 316)), (151, (488, 328)), (152, (516, 342)), (153, (546, 360)), (154, (571, 377)), 
                               (155, (597, 398)), (156, (616, 396)), (157, (631, 375)), (168, (629, 256)), (169, (599, 245)), 
                               (170, (570, 237)), (171, (540, 231)), (173, (482, 227)), (174, (454, 229)), (175, (425, 233)), 
                               (176, (397, 240)), (177, (369, 249)), (178, (341, 261)), (179, (314, 276)), (180, (809, 363)), 
                               (180, (285, 292)), (181, (258, 312)), (182, (230, 335)), (183, (203, 359)), (184, (176, 386)), 
                               (204, (152, 250)), (205, (193, 237)), (206, (235, 227)), (207, (273, 221)), (208, (310, 217)), 
                               (209, (347, 217)), (210, (382, 218)), (211, (416, 223)), (212, (449, 231)), (213, (481, 242)), 
                               (214, (510, 254)), (215, (538, 268)), (220, (669, 373)), (243, (465, 178)), (244, (440, 190)), 
                               (245, (415, 204)), (246, (389, 221)), (247, (364, 240)), (248, (339, 262)), (249, (314, 288)), 
                               (250, (288, 315)), (251, (262, 346)), (252, (236, 379)), (253, (210, 415)), (254, (189, 440)), 
                               (255, (177, 402)), (256, (166, 375)), (257, (155, 353)), (266, (285, 243)), (267, (325, 241)), 
                               (268, (366, 243)), (269, (405, 247)), (270, (442, 253)), (271, (478, 262)), (274, (581, 306)), 
                               (276, (641, 343)), (277, (673, 368)), (278, (701, 392)), (279, (729, 422)), (280, (747, 393)), 
                               (281, (763, 370)), (282, (779, 351)), (283, (795, 333)), (291, (754, 220)), (320, (344, 352))]
"""

# A function to perform linear interpolation between two points
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

# def find_closest_histogram(image_hist, histogram_array):
#     closest_index = -1
#     min_distance = float('inf')

#     for i, hist in enumerate(histogram_array):
#         # Calculate the Bhattacharyya distance between the image's histogram and each histogram in the array
#         distance = cv2.compareHist(image_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

#         # Check if the current histogram is closer than the previous closest one
#         if distance < min_distance:
#             min_distance = distance
#             closest_index = i

#     # return closest_index, min_distance
#     return closest_index

def is_player(contour_hist,p1_hist,p2_hist):
    distance_p1 = cv2.compareHist(contour_hist, p1_hist, cv2.HISTCMP_BHATTACHARYYA)
    distance_p2 = cv2.compareHist(contour_hist, p2_hist, cv2.HISTCMP_BHATTACHARYYA) 

    min_distance = float(0.94)
    if distance_p1 < min_distance or distance_p2 < min_distance:


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

        # Check if there has been a change in vertical direction to "up"
        if current_vertical_direction == "up" and vertical_direction == "down":
            direction_changes.append((current_frame, "BOUNCE"))

        # Check if there has been a change in horizontal direction
        if current_horizontal_direction != horizontal_direction and horizontal_direction is not None:
            direction_changes.append((current_frame, "PADDLE"))
        
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
                    cv2.circle(ball_path, (cX, cY), 5, (255, 0, 0), -1)
                    ball_locations.append((frame_number,(cX,cY)))
                else: 
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


    # Display the original frame and the result
    if show: cv2.imshow('Original Video', frame)
    if show: cv2.imshow('Ball Path', ball_path)

    if cv2.waitKey(30) & 0xFF == 27: break  # Press 'ESC' button to exit the while loop


cap.release()
cv2.destroyAllWindows()

filled_gaps_locations = fill_gaps(ball_locations)
track_trajectory(filled_gaps_locations)

cv2.imshow("filled coords", draw_path(filled_gaps_locations))

# print(filled_coords)

cv2.waitKey(0)
cv2.destroyAllWindows()