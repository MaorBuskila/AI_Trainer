import cv2
import PoseModule as pm
import numpy as np
# Load the video
def edgeDetector(image):
  '''
  This function should get as input the grayscale 'image' and any additional
  parameters you need, and return 'edge_map': a binary image (same shape as 'image')
  with a value of 1 in each detected edge pixel and a value of zero otherwise.
  '''
  t1 = 150
  t2 = 2 * t1
  edge_map = cv2.Canny(image, threshold1=t1, threshold2=t2) # Replace with edge detection code
  return edge_map

def draw_retrrangle_arm(x12,y12,x14,y14):


    arm_bbox = (x14, y12, abs(x14-x12) + 20, abs(y12-y14)+20)  # (x, y, w, h) of the bounding box

    # Draw the bounding box
    cv2.rectangle(frame, (arm_bbox[0], arm_bbox[1]), (arm_bbox[0] + arm_bbox[2], arm_bbox[1] + arm_bbox[3]),
                  (0, 255, 0), 2)


# Step 1: Produce an edge map from the image using an edge detector

cap = cv2.VideoCapture('video.mp4')
detector = pm.poseDetector()

# Loop through the frames
while True:
    # Read the frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    frame = detector.findPose(frame, False)
    lmList = detector.findPosition(frame, False)
    if len(lmList) != 0:
            # Right Arm
        angle = detector.findAngle(frame, 12, 14, 16)
        x12, y12 = lmList[12][1:]
        x14, y14 = lmList[14][1:]
        x16, y16 = lmList[16][1:]
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # draw_retrrangle_arm(x12,y12,x16,y16)
    lower_blue = np.array([128, 86, 75])
    upper_blue = np.array([210, 190, 182])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Check if there was a frame to read
    if not ret:
        break

    # Convert the image to grayscale and apply a threshold to create a binary mask that separates the objects from the background.
    # The threshold value can be manually set or calculated automatically using a method
    # like Otsu's thresholding

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Apply morphological operations like dilation or erosion
    # to the binary mask to remove noise or close gaps between the objects

    # Apply filters and other preprocessing steps as necessary

    # Detect the arm
    # Replace this with your own code to detect the arm
    # arm_bbox = (100, 100, 200, 200)  # (x, y, w, h) of the bounding box

    # Draw the bounding box
    # cv2.rectangle(frame, (arm_bbox[0], arm_bbox[1]), (arm_bbox[0] + arm_bbox[2], arm_bbox[1] + arm_bbox[3]),
    #               (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Frame', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video and close the window
cap.release()
cv2.destroyAllWindows()
