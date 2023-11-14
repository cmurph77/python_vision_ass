import cv2

# Open the video file
video_path = "TableTennis.avi"
cap = cv2.VideoCapture(video_path)
num = 0;

if not cap.isOpened():
    print("Error: Could not open the video file.")
    exit()

while True:
    num = num + 1
    print(num)
    ret, frame = cap.read()
    if not ret:
        break

    # Perform frame analysis here

    # Display the frame
    # cv2.imshow('Frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("frame count", num)
cap.release()
cv2.destroyAllWindows()
