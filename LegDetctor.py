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
    y = y12 -50
    w = max(abs(x16-x12)+ 100, abs(x14-x12)+ 100)
    h = max(abs(y12 - y14)+100,abs(y12 - y16)+100)
    return x, y, w, h

def crop_frame(frame,x,y,w,h):
    # Draw the bounding box
    cv2.rectangle(frame, (x,y,w,h),
                  (0, 255, 0), 2)
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

    x, y, w, z = calculate_arm_position(x12, y12, x14, y14, x16, y16)
    cropped = crop_frame(frame,x,y,w,z)
    # blurred_frame = cv2.GaussianBlur(cropped, (5, 5), 0)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)

    low = np.array([0, 42, 0])
    high = np.array([255, 255, 179])
    mask = cv2.inRange(hsv, low, high)
    # Check if there was a frame to read
    if not ret:
        break

    # Convert the image to grayscale and apply a threshold to create a binary mask that separates the objects from the background.
    # The threshold value can be manually set or calculated automatically using a method
    # like Otsu's thresholding

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    # _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # Apply morphological operations like dilation or erosion
    # to the binary mask to remove noise or close gaps between the objects

    # cv2.imshow('Frame',cropped )
    cv2.imshow('Frame',mask )

    # Exit if the user presses 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video and close the window
cap.release()
cv2.destroyAllWindows()
