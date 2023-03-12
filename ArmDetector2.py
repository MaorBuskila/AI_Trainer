import cv2
import numpy as np
import PoseModule as pm
class armDetctor2():

    detector = pm.poseDetector()
    def detect(self,frame,detector):
        while True:
            frame = cv2.resize(frame, (1280, 720))
            frame = detector.findPose(frame, False)
            lmList = detector.findPosition(frame, False)
            if len(lmList) != 0:
                    # Right Arm
                angle = detector.findAngle(frame, 12, 14, 16,False)
                # Extract arm region
                x12, y12 = lmList[12][1:]
                x14, y14 = lmList[14][1:]
                x16, y16 = lmList[16][1:]
            # Apply skin color filter
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            masked_img = cv2.bitwise_and(frame, frame, mask=mask)



            x_min = min(x12, x14, x16)-25
            x_max = max(x12, x14, x16)+25
            y_min = min(y12, y14, y16)-25
            y_max = max(y12, y14, y16)+25
            arm_img = masked_img[y_min:y_max, x_min:x_max]


            # Extract arm contour
            gray = cv2.cvtColor(arm_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original image
            cv2.drawContours(frame[y_min:y_max, x_min:x_max], contours, -1, (0, 0, 255), 2)
            # Check if there was a frame to read
            if not ret:
              break
            cv2.imshow('Frame', frame)

            # Exit if the user presses 'q'
            if cv2.waitKey(1) == ord('q'):
              break

    # Release the video and close the window
    cap.release()
    cv2.destroyAllWindows()

