import math
import cv2 
import numpy as np

def print_array(a):
    
    video_height = 540
    video_width = 960
    path_img = np.zeros((video_height, video_width, 3), dtype=np.uint8)
    prev_center = (0,0)
    first = True
    for  point in a:
        current_center = point[1]
        print(first)
        if first: 
            first = False
            print("drawing lines", first)
        else: 
            print("not first")
            cv2.line(path_img, prev_center,current_center,(255,0,0),2)

        cv2.circle(path_img, current_center, 3, (0,0,255), -1)

        prev_center = current_center
    return path_img

ball_locations = [(21, (667, 408)), (22, (639, 386)), (23, (612, 369)), (24, (766, 367)), (25, (769, 368)),
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

# Function to calculate the Euclidean distance between two points
def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


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

def remove_duplicated(ball_locations):
    # Processed list initialization
    processed_locations = []

    # Initialize previous location with the first coordinate
    prev_location = ball_locations[0][1]

    for i in range(len(ball_locations)):
        frame, coord = ball_locations[i]
        
        # Check if we're at the last frame or the next frame is different
        if i == len(ball_locations) - 1 or ball_locations[i][0] != ball_locations[i + 1][0]:
            processed_locations.append((frame, coord))
            prev_location = coord
        else:
            # Calculate distance to the previous location
            dist = distance(prev_location, coord)
            # Look ahead at the next frames with the same frame number
            j = i + 1
            while j < len(ball_locations) and ball_locations[j][0] == frame:
                next_dist = distance(prev_location, ball_locations[j][1])
                if next_dist < dist:
                    coord = ball_locations[j][1]
                    dist = next_dist
                j += 1
            # Append the closest coordinate to the processed list
            processed_locations.append((frame, coord))
            prev_location = coord
            # Skip the frames that we have already processed
            i = j - 1
        return processed_locations

cv2.imshow("initial ball path", print_array(ball_locations))

no_duplicates = remove_duplicated(ball_locations)

cv2.imshow("Duplicates Removed", print_array(no_duplicates))

filled_coords = fill_gaps(ball_locations)
cv2.imshow("filled coords", print_array(filled_coords))

# print(filled_coords)

cv2.waitKey(0)
cv2.destroyAllWindows()

