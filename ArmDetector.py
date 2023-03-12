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
  t1 = 50
  t2 = 100
  edge_map = cv2.Canny(image, threshold1=t1, threshold2=t2) # Replace with edge detection code
  return edge_map

def calculate_arm_position(x12, y12 ,x14, y14 ,x16, y16 ):
    x = x12 - 50
    y = y12
    w = max(abs(x16-x12)+ 75, abs(x14-x12)+ 75)
    h = max(abs(y12 - y14)+75,abs(y12 - y16)+75)
    return x, y, w, h

def crop_frame(frame,x,y,w,h):
    # Draw the bounding box
    cropped_image = frame[y:y + h, x:x+w]
    return cropped_image



# Step 1: Produce an edge map from the image using an edge detector

cap = cv2.VideoCapture('Hammer.mp4')
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
        angle = detector.findAngle(frame, 12, 14, 16,False)
        x12, y12 = lmList[12][1:]
        x14, y14 = lmList[14][1:]
        x16, y16 = lmList[16][1:]

    x, y, width, height = calculate_arm_position(x12, y12, x14, y14, x16, y16)
    roi = crop_frame(frame, x, y, width, height)
    rgb_lower = np.array([64, 76, 87])
    rgb_upper = np.array([163, 164, 226])
    mask = cv2.inRange(roi, rgb_lower, rgb_upper)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    inv_mask = cv2.bitwise_not(mask)

    h, s, v = cv2.split(hsv)
    h = np.mod(h + 50, 180)
    s = np.clip(s - 0, 0, 255)
    v = np.clip(v + 20, 0, 255)
    hsv = cv2.merge([h, s, v])

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    result = cv2.bitwise_or(cv2.bitwise_and(roi, roi, mask=inv_mask), cv2.bitwise_and(bgr, bgr, mask=mask))
    alpha = 0.5
    blended = cv2.addWeighted(roi, alpha, result, 1 - alpha, 0)
    frame[y:y + height, x:x + width] = blended

    # Check if there was a frame to read
    if not ret:
        break

    cv2.imshow('Frame',frame )

    # Exit if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video and close the window
cap.release()
cv2.destroyAllWindows()
