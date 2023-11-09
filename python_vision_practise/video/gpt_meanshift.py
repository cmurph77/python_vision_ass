import cv2

# Load the video
video_path = '../TableTennis.avi'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Create a tracker (Mean-Shift in this case)
tracker = cv2.TrackerCSRT_create()


# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error reading video file.")
    exit()

for i in range(1, 30):
    ret, frame = cap.read()
    if not ret:
        break


ball_img = cv2.imread("balls/Ball5")

# Define a region of interest (ROI) around the ball in the first frame
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

# Loop through the video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker with the new frame
    success, bbox = tracker.update(frame)

    if success:
        # Tracking success - draw a bounding box around the ball
        (x, y, w, h) = [int(i) for i in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failed", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the current frame
    cv2.imshow('Tennis Ball Tracking', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
